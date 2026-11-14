"""Optional web search before each generation step, for single and swarm mode alike.

When a run has web search on, every step that writes new hypotheses (a single-mode generation call; each swarm
worker) first asks the model for up to `n_queries` short search queries, runs them, and puts the results into that
step's prompt as a WEB SEARCH RESULTS block: the engine's title, URL and snippet for the top `k` hits of each query,
plus a short excerpt from the first `fetch_pages` pages, cut to a character budget. Nothing else changes, so a run
with search off sends exactly the prompts it sent before.

Results are cached on disk by query (and pages by URL). Runs that share a cache file see the same results for the
same query, which keeps a comparison across models or modes from depending on when each run happened to search.
Every query, hit and excerpt a step saw is kept in `WebSearch.log`, so a reader can check where a claim came from.
"""

from __future__ import annotations

import datetime
import html as H
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

PREAMBLE = (
    "WEB SEARCH RESULTS - the results of the web searches you asked for: search-engine snippets and short page "
    "excerpts, retrieved {date}. They can be incomplete or wrong. Cite the URL for anything you take from them, and keep "
    "numbers from the web apart from numbers in any attached data."
)
_TAG = re.compile(r"<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", re.S | re.I)
_WORD = re.compile(r"[a-z][a-z0-9]{2,}")


def _page_text(raw_html: str) -> str:
    t = _TAG.sub(" ", raw_html)
    t = re.sub(r"<(br|p|div|li|h[1-6]|tr)[^>]*>", "\n", t, flags=re.I)
    t = H.unescape(re.sub(r"<[^>]+>", " ", t))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in t.splitlines()]
    return "\n".join(ln for ln in lines if len(ln) > 40)


def _excerpt(text: str, query: str, n: int) -> str:
    """The paragraphs that share the most words with the query, in page order, up to n characters."""
    q = set(_WORD.findall(query.lower()))
    paras = [p for p in text.split("\n") if p]
    scored = sorted(range(len(paras)), key=lambda i: -len(q & set(_WORD.findall(paras[i].lower()))))
    keep, used = set(), 0
    for i in scored:
        if not q & set(_WORD.findall(paras[i].lower())):
            break
        if used + len(paras[i]) > n:
            continue
        keep.add(i)
        used += len(paras[i]) + 1
    return "\n".join(paras[i] for i in sorted(keep))


def default_searcher(query: str, k: int, backend: str) -> List[Dict[str, str]]:
    from ddgs import DDGS  # optional dependency; imported only when web search is used

    hits = DDGS().text(query, max_results=k, backend=backend) or []
    return [dict(title=h.get("title", ""), url=h.get("href", ""), snippet=h.get("body", "")) for h in hits]


def default_fetcher(url: str) -> str:
    import requests

    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (research-assistant search step)"})
    if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
        return ""
    return _page_text(r.text[:2_000_000])


def parse_queries(reply: str, n: int) -> List[str]:
    """A JSON array of strings if the reply has one, otherwise one query per non-empty line."""
    m = re.search(r"\[.*\]", reply or "", re.S)
    qs: List[Any] = []
    if m:
        try:
            qs = json.loads(m.group(0))
        except ValueError:
            qs = []
    if not qs:
        qs = [re.sub(r"^[\s\-*\d.)\"']+|[\"',]+$", "", ln) for ln in (reply or "").splitlines()]
    out = []
    for q in qs:
        q = re.sub(r"\s+", " ", str(q)).strip()[:200]
        if q and q.lower() not in {x.lower() for x in out} and not q.lower().startswith("error"):
            out.append(q)
    return out[:n]


class WebSearch:
    def __init__(
        self,
        cache_path: Optional[str] = None,
        n_queries: int = 3,
        k: int = 5,
        fetch_pages: int = 2,
        budget: int = 6000,
        backends: Sequence[str] = ("bing", "bing", "auto", "auto"),
        searcher: Optional[Callable[[str, int, str], List[Dict[str, str]]]] = None,
        fetcher: Optional[Callable[[str], str]] = None,
        wait: float = 2.0,
    ):
        self.cache_path, self.n_queries, self.k, self.fetch_pages, self.budget = (
            cache_path,
            n_queries,
            k,
            fetch_pages,
            budget,
        )
        self.backends, self.wait = list(backends), wait
        self.searcher = searcher or default_searcher
        self.fetcher = fetcher or default_fetcher
        self.log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._search_lock = threading.Lock()  # one search at a time: engines refuse bursts
        self._cache: Dict[str, Any] = {"queries": {}, "pages": {}}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def describe(self) -> str:
        return (
            f"web search before each generation step: up to {self.n_queries} model-written queries, top {self.k} hits "
            f"each (engine: {self.backends[0]}, then the others in {self.backends} on failure), excerpts of the first {self.fetch_pages} pages, about {self.budget:,} characters"
        )

    def _save(self) -> None:
        if self.cache_path:
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._cache, f, indent=1)
            os.replace(tmp, self.cache_path)

    def search(self, query: str) -> Dict[str, Any]:
        key = f"{self.k}|{query.strip().lower()}"
        with self._lock:
            hit = self._cache["queries"].get(key)
        if hit is not None:
            return dict(hit, cached=True)
        hits, err, used = [], None, None
        with self._search_lock:
            for i, b in enumerate(self.backends):  # an engine that refuses or finds nothing is retried, then the next
                try:
                    hits, err, used = self.searcher(query, self.k, b), None, b
                    if hits:
                        break
                    err = "no results"
                except Exception as e:  # network or engine error: the step goes on without these results
                    err = f"{type(e).__name__}: {str(e)[:200]}"
                if i + 1 < len(self.backends):
                    time.sleep(self.wait)
        for h in hits[: self.fetch_pages]:
            url = h.get("url", "")
            with self._lock:
                page = self._cache["pages"].get(url)
            if page is None:
                try:
                    page = self.fetcher(url) if url else ""
                except Exception:
                    page = ""
                with self._lock:
                    self._cache["pages"][url] = page
            h["excerpt"] = _excerpt(page, query, 900)
        rec = dict(query=query, hits=hits, error=err, engine=used, retrieved=datetime.date.today().isoformat())
        with self._lock:
            if err is None:  # failures are not cached, so a later run can try again
                self._cache["queries"][key] = rec
                self._save()
        return dict(rec, cached=False)

    def queries_prompt(self, goal: str, direction: str = "") -> str:
        return (
            f"Research Goal: {goal}\n\n"
            + (f"Your direction: {direction}\n\n" if direction else "")
            + f"Before you propose hypotheses you can search the web. Write up to {self.n_queries} short web search "
            "queries (a few words each) whose results would help you propose and test hypotheses for this goal"
            + (" in your direction" if direction else "")
            + ". Return only a JSON array of strings."
        )

    def block(self, results: List[Dict[str, Any]]) -> str:
        date = max((r.get("retrieved", "") for r in results), default="") or datetime.date.today().isoformat()
        out, used = [PREAMBLE.format(date=date)], 0
        for r in results:
            lines = [f'### Search: "{r["query"]}"']
            if not r["hits"]:
                lines.append("(no results)")
            for i, h in enumerate(r["hits"], 1):
                lines.append(f"{i}. {h.get('title', '')} - {h.get('url', '')}\n   {h.get('snippet', '')}")
                if h.get("excerpt"):
                    lines.append("   Page excerpt: " + h["excerpt"].replace("\n", " "))
            part = "\n".join(lines)
            if used + len(part) > self.budget:
                part = part[: max(0, self.budget - used)] + "\n[... cut to fit ...]"
            out.append(part)
            used += len(part)
            if used >= self.budget:
                break
        return "\n\n".join(out)

    def step(self, ask: Callable[[str], str], goal: str, direction: str = "", step: str = "") -> str:
        """Ask for queries, run them, log everything, and return the block for the step's prompt ('' if none ran)."""
        prompt = self.queries_prompt(goal, direction)
        reply = ask(prompt) or ""
        queries = parse_queries(reply, self.n_queries)
        results = [self.search(q) for q in queries]
        text = self.block(results) if results else ""
        with self._lock:
            self.log.append(dict(step=step, direction=direction, reply=reply, queries=queries, results=results))
        return ("\n\n" + text + "\n\n") if text else ""
