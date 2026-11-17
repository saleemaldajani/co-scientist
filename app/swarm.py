"""Swarm mode: many co-scientist workers on one goal, pooled and ranked by LLM debate.

The default cycle (agents.SupervisorAgent) asks one generation call for hypotheses, ranks them by the
HIGH/MEDIUM/LOW labels of the reflection step, "evolves" by concatenating the top two texts and writes a
templated meta-review. Swarm mode keeps those agents' roles but gives each one an LLM, following published
multi-agent designs:

- Orchestrator-worker split (multi-agent research systems): an orchestrator splits the goal into
  distinct research directions, each with an objective and boundaries; one worker per direction generates
  hypotheses in parallel.
- Routines and handoffs (agent SDK pattern): every role is a routine (fixed instructions and a JSON
  output); work moves between routines by handoff, recorded in the trace.
- Tournament of pairwise debates with Elo, proximity de-duplication, evolution of the top ideas from their
  critiques, and a meta-review whose guidance feeds the next round (AI co-scientist designs).
- Debate between agents (Du et al. 2023, multi-agent debate): the judge weighs the strongest argument for and
  against each hypothesis before choosing.

Every LLM call is recorded (role, seconds, prompt and response sizes) in the returned trace.

Uploaded data (optional, app.data_context.DataBundle): every step gets an extract of the files chosen for that step
(the goal for the orchestrator, the worker's direction, the two hypotheses a judge compares, the hypothesis being
rewritten), and every hypothesis's numbers are checked against the files at the end.
"""

import json
import math
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .agents import call_llm_for_reflection
from .data_context import DataBundle
from .science_tools import ScienceTools
from .utils import call_llm, last_model_used, last_usage, logger, similarity_score
from .web_search import WebSearch

DEFAULT_LENSES = [
    {
        "name": "Mechanism",
        "objective": "Hypotheses about the causal mechanism behind the pattern the goal asks about.",
        "boundaries": "Avoid purely methodological ideas.",
    },
    {
        "name": "Measurement",
        "objective": "Hypotheses about what a particular measurement, record or data source reveals that others cannot.",
        "boundaries": "Each must hinge on the measurement itself.",
    },
    {
        "name": "Population",
        "objective": "Hypotheses about who is affected, when, and how outcomes differ across groups.",
        "boundaries": "Avoid single-mechanism ideas.",
    },
    {
        "name": "Contrarian",
        "objective": "Hypotheses that challenge an accepted finding or use the data in an unexpected way.",
        "boundaries": "Must still be testable with the stated data.",
    },
]


def _strip_json(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _salvage_objects(t: str) -> List[Any]:
    """Complete JSON objects from an array cut off mid-way (a reply that ran out of output tokens)."""
    i = t.find("[")
    if i < 0:
        return []
    dec, pos, out = json.JSONDecoder(), i + 1, []
    while True:
        while pos < len(t) and t[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(t) or t[pos] != "{":
            return out
        try:
            obj, pos = dec.raw_decode(t, pos)
        except json.JSONDecodeError:
            try:
                obj, pos = json.JSONDecoder(strict=False).raw_decode(t, pos)
            except json.JSONDecodeError:
                return out
        out.append(obj)


def parse_json(text: str, want: type):
    """Parse a JSON object or array from an LLM reply; tolerate fences, leading or trailing prose, control
    characters inside strings, and (for arrays) a reply cut off after some complete objects."""
    v = _parse_json(text, want)
    if v is None and want is list:
        v = _salvage_objects(_strip_json(text)) or None
    return v


def _parse_json(text: str, want: type):
    t = _strip_json(text)
    try:
        v = json.loads(t)
        if isinstance(v, want):
            return v
    except (json.JSONDecodeError, ValueError):
        pass
    o, c = ("[", "]") if want is list else ("{", "}")
    i, j = t.find(o), t.rfind(c)
    if i >= 0 and j > i:
        for strict in (True, False):
            try:
                v = json.loads(t[i : j + 1], strict=strict)
                if isinstance(v, want):
                    return v
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def reply_snippet(text: str, n: int = 140) -> str:
    t = " ".join((text or "").split())
    return "empty reply" if not t else f"reply began: {t[:n]!r}"


def as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return "\n\n".join(f"**{k}**: {as_text(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "\n".join(as_text(x) for x in v)
    return str(v)


def expected(ra: float, rb: float) -> float:
    return 1 / (1 + math.pow(10, (rb - ra) / 400))


class SwarmCoScientist:
    def __init__(
        self,
        goal: str,
        model: Optional[str] = None,
        n_workers: int = 4,
        per_worker: int = 10,
        rounds: int = 2,
        matches_per_round: int = 3,
        top_k_evolve: int = 5,
        final_k: int = 10,
        dedupe_threshold: float = 0.85,
        parallel: int = 4,
        k_factor: int = 32,
        generation_temperature: float = 0.8,
        judge_temperature: float = 0.2,
        seed: int = 0,
        progress: Optional[Callable[[str], None]] = None,
        data: Optional[DataBundle] = None,
        data_budget: int = 20000,
        web: Optional[WebSearch] = None,
        tools: Optional[ScienceTools] = None,
    ):
        self.goal, self.model = goal, model
        self.n_workers, self.per_worker, self.rounds = n_workers, per_worker, rounds
        self.matches_per_round, self.top_k_evolve, self.final_k = matches_per_round, top_k_evolve, final_k
        self.dedupe_threshold, self.parallel, self.k = dedupe_threshold, parallel, k_factor
        self.gen_t, self.judge_t = generation_temperature, judge_temperature
        self.rng = random.Random(seed)
        self.progress = progress or (lambda m: logger.info("[swarm] %s", m))
        self.trace: List[Dict[str, Any]] = []
        self.hyps: Dict[str, Dict[str, Any]] = {}
        self.matches: List[Dict[str, Any]] = []
        self.meta_reviews: List[Dict[str, Any]] = []
        self.merged: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self._n = 0
        self._lock = threading.Lock()
        self.data, self.data_budget = (data if data else None), data_budget
        self.web = web
        self.tools = tools if tools else None

    # ------------------------------------------------------------------ plumbing
    def llm(self, role: str, prompt: str, temperature: float) -> str:
        t0 = time.time()
        out = call_llm(prompt, temperature=temperature, model=self.model)
        self.trace.append(
            dict(
                role=role,
                model_used=last_model_used() if not (out or "").startswith("Error:") else None,
                seconds=round(time.time() - t0, 1),
                prompt_chars=len(prompt),
                response_chars=len(out or ""),
                error=(out or "").startswith("Error:"),
                prompt=prompt,
                response=out,
                **last_usage(),
            )
        )
        if (out or "").startswith("Error:"):
            self.errors.append(f"{role}: {out[:300]}")
        return out or ""

    def ask_json(self, role: str, prompt: str, temperature: float, want: type):
        """Call and parse; if the reply cannot be read, ask once more for the bare JSON. Returns (value, last reply)."""
        out = self.llm(role, prompt, temperature)
        v = parse_json(out, want)
        if v is None and not out.startswith("Error:"):
            kind = "array" if want is list else "object"
            out = self.llm(
                role + " (retry)",
                prompt
                + f"\n\nYour previous reply could not be read as JSON. Return ONLY the JSON {kind}, with no other text.",
                temperature,
            )
            v = parse_json(out, want)
        return v, out

    def pmap(self, fn, items):
        with ThreadPoolExecutor(max_workers=max(1, self.parallel)) as ex:
            return list(ex.map(fn, items))

    def new_id(self, prefix: str) -> str:
        with self._lock:
            self._n += 1
            return f"{prefix}{self._n:03d}"

    def goal_brief(self, n: int = 6000) -> str:
        return self.goal if len(self.goal) <= n else self.goal[:n] + "\n[... goal truncated for this step ...]"

    def data_block(self, query: str, budget: Optional[int] = None) -> str:
        """Extract of the uploaded files for one step ('' without data, so prompts are unchanged)."""
        if not self.data:
            return ""
        return "\n\n" + self.data.brief(query, budget or self.data_budget) + "\n\n"

    def web_block(self, w: int, lens: Dict[str, str]) -> str:
        """Web search results for worker w ('' with search off, so prompts are unchanged)."""
        if not self.web:
            return ""
        direction = f"{lens.get('name', '')}: {lens.get('objective', '')}"
        return self.web.step(
            lambda p: self.llm(f"worker{w + 1} search", p, self.gen_t), self.goal, direction, step=f"worker {w + 1}"
        )

    def tools_block(self, w: int, lens: Dict[str, str]) -> str:
        """Science tool output for worker w ('' with tools off, so prompts are unchanged)."""
        if not self.tools:
            return ""
        direction = f"{lens.get('name', '')}: {lens.get('objective', '')}"
        return self.tools.step(
            lambda p: self.llm(f"worker{w + 1} tools", p, self.gen_t), self.goal, direction, step=f"worker {w + 1}"
        )

    # ------------------------------------------------------------------ routines
    def orchestrate(self) -> List[Dict[str, str]]:
        prompt = (
            "You are the orchestrator of a swarm of AI co-scientists. Split the research goal below into "
            f"{self.n_workers} distinct, non-overlapping research directions, one per worker. Each direction needs a short "
            "name, an objective (what kind of hypotheses to look for), the data types it should favor, and boundaries "
            "(what to leave to the other workers). Together the directions must cover the goal.\n\n"
            f"RESEARCH GOAL:\n{self.goal_brief()}\n\n"
            + self.data_block(self.goal, min(self.data_budget, 8000))
            + 'Return only a JSON array of objects with keys "name", "objective", "data_types", "boundaries".'
        )
        out, reply = self.ask_json("orchestrator", prompt, 0.5, list)
        lenses = [x for x in (out or []) if isinstance(x, dict) and x.get("objective")][: self.n_workers]
        if len(lenses) < self.n_workers:
            self.errors.append(f"orchestrator: fell back to default directions ({reply_snippet(reply)})")
            lenses = (lenses + DEFAULT_LENSES)[: self.n_workers]
        return [{k: as_text(v) for k, v in lens.items()} for lens in lenses]

    def work(self, args) -> List[Dict[str, Any]]:
        w, lens = args
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            f"You are worker {w + 1} of {self.n_workers} in a swarm of co-scientists. Your direction: {lens.get('name', '')}.\n"
            f"Objective: {lens.get('objective', '')}\nFavor data types: {lens.get('data_types', '')}\n"
            f"Boundaries (left to other workers): {lens.get('boundaries', '')}\n\n"
            f"Propose {self.per_worker} novel, testable hypotheses within your direction, each complete in the format the goal asks for.\n\n"
            + self.data_block(
                f"{self.goal[:3000]} {lens.get('name', '')} {lens.get('objective', '')} {lens.get('data_types', '')}"
            )
            + self.web_block(w, lens)
            + self.tools_block(w, lens)
            + "Please return the response as a JSON array of objects, where each object has a 'title' and 'text' key."
        )
        data, reply = self.ask_json(f"worker{w + 1}", prompt, self.gen_t, list)
        data = data or []
        out = []
        for d in data:
            if isinstance(d, dict) and d.get("title") and d.get("text"):
                out.append(
                    dict(
                        id=self.new_id("S"),
                        title=as_text(d["title"]),
                        text=as_text(d["text"]),
                        worker=w + 1,
                        lens=lens.get("name", ""),
                        round=1,
                        parent_ids=[],
                        elo=1200.0,
                        wins=0,
                        losses=0,
                        review_comments=[],
                        references=[],
                        active=True,
                    )
                )
        if not out:
            self.errors.append(f"worker{w + 1}: no parsable hypotheses ({reply_snippet(reply)})")
        return out

    def dedupe(self, new: List[Dict[str, Any]]) -> None:
        """Proximity step: a hypothesis too similar to one already kept is merged into it (kept inactive, recorded)."""
        kept = [h for h in self.hyps.values() if h["active"]]
        for h in new:
            self.hyps[h["id"]] = h
            best, sim = None, 0.0
            for k in kept:
                s = similarity_score(h["title"] + "\n" + h["text"][:1500], k["title"] + "\n" + k["text"][:1500])
                if s > sim:
                    best, sim = k, s
            if best is not None and sim >= self.dedupe_threshold:
                h["active"] = False
                self.merged.append(dict(merged=h["id"], into=best["id"], similarity=round(sim, 3)))
            else:
                kept.append(h)

    def reflect(self, h: Dict[str, Any]) -> None:
        t0 = time.time()
        r = call_llm_for_reflection(h["text"], temperature=0.5, model=self.model)
        self.trace.append(
            dict(
                role="reflection",
                seconds=round(time.time() - t0, 1),
                prompt_chars=len(h["text"]),
                response_chars=len(json.dumps(r)),
                error=str(r.get("comment", "")).startswith(("Could not parse", "LLM review failed")),
                prompt="",
                response=json.dumps(r),
            )
        )
        h["novelty_review"], h["feasibility_review"] = r.get("novelty_review"), r.get("feasibility_review")
        h["review_comments"].append(as_text(r.get("comment", "")))
        h["references"] += [as_text(x) for x in r.get("references", []) if as_text(x) not in h["references"]]

    def judge(self, pair) -> Dict[str, Any]:
        a, b = pair
        if self.rng.random() < 0.5:  # randomize position to limit order bias
            a, b = b, a
        prompt = (
            "You are a panel of expert reviewers in a hypothesis tournament. Compare two hypotheses for the research goal.\n"
            "Criteria: fit to the goal; novelty; whether it can be tested with the data the goal describes; whether the data "
            "names, cohort sizes and numbers it quotes are consistent with the goal's information; falsifiability; expected impact.\n"
            "Debate briefly: give the strongest argument for and against each, then decide.\n\n"
            f"RESEARCH GOAL (summary):\n{self.goal_brief(4000)}\n\n"
            f"HYPOTHESIS A: {a['title']}\n{a['text'][:6000]}\n\nHYPOTHESIS B: {b['title']}\n{b['text'][:6000]}\n\n"
            + (
                self.data_block(
                    f"{a['title']} {a['text'][:3000]} {b['title']} {b['text'][:3000]}", min(self.data_budget, 8000)
                )
                + "Check the numbers each hypothesis quotes against the attached data; a number that is not there counts against it.\n\n"
                if self.data
                else ""
            )
            + 'Return only a JSON object: {"for_A": "...", "against_A": "...", "for_B": "...", "against_B": "...", "winner": "A" or "B", "reason": "..."}'
        )
        d = self.ask_json("judge", prompt, self.judge_t, dict)[0] or {}
        w = str(d.get("winner", "")).strip().upper()[:1]
        if w not in ("A", "B"):
            w = self.rng.choice(["A", "B"])
            d["reason"] = "judge reply not parsable: random winner"
            self.errors.append("judge: unparsable verdict")
        win, lose = (a, b) if w == "A" else (b, a)
        return dict(
            winner=win["id"],
            loser=lose["id"],
            reason=as_text(d.get("reason", "")),
            against_loser=as_text(d.get(f"against_{'B' if w == 'A' else 'A'}", "")),
        )

    def tournament(self, round_no: int) -> None:
        """Swiss-style: each round pairs neighbors in the current Elo order; matches_per_round rounds."""
        for m in range(self.matches_per_round):
            act = sorted([h for h in self.hyps.values() if h["active"]], key=lambda h: -h["elo"])
            if m == 0 and round_no == 1:
                self.rng.shuffle(act)
            pairs = [(act[i], act[i + 1]) for i in range(0, len(act) - 1, 2)]
            for res in self.pmap(self.judge, pairs):
                win, lose = self.hyps[res["winner"]], self.hyps[res["loser"]]
                ea = expected(win["elo"], lose["elo"])
                win["elo"] += self.k * (1 - ea)
                lose["elo"] -= self.k * (1 - ea)
                win["wins"] += 1
                lose["losses"] += 1
                self.matches.append(dict(res, round=round_no, match=m + 1))

    def meta_review(self, round_no: int) -> Dict[str, Any]:
        act = sorted([h for h in self.hyps.values() if h["active"]], key=lambda h: -h["elo"])
        lines = [
            f"#{k + 1} (Elo {h['elo']:.0f}) {h['title']} | review: {h['review_comments'][-1][:300] if h['review_comments'] else ''}"
            for k, h in enumerate(act)
        ]
        reasons = [m["reason"][:250] for m in self.matches if m["round"] == round_no][:30]
        prompt = (
            "You are the meta-reviewer of a co-scientist swarm. Read the ranked hypotheses, their reviews and the reasons "
            "judges gave in the tournament. Identify recurring weaknesses (for example wrong data names, unsupported cohort "
            "sizes, weak designs, missing controls) and strengths, and give concrete guidance for improving the top hypotheses.\n\n"
            f"RESEARCH GOAL (summary):\n{self.goal_brief(3000)}\n\nRANKED HYPOTHESES:\n"
            + "\n".join(lines[:40])
            + "\n\nJUDGES' REASONS:\n"
            + "\n".join(f"- {r}" for r in reasons)
            + '\n\nReturn only a JSON object: {"critique": ["..."], "strengths": ["..."], "guidance": ["..."]}'
        )
        d = self.ask_json("meta_review", prompt, 0.4, dict)[0] or {"critique": [], "strengths": [], "guidance": []}
        d = {k: [as_text(x) for x in (v if isinstance(v, list) else [v])] for k, v in d.items()}
        d["round"] = round_no
        self.meta_reviews.append(d)
        return d

    def evolve(self, args) -> Optional[Dict[str, Any]]:
        h, guidance = args
        losses = [m["against_loser"] for m in self.matches if m["loser"] == h["id"] and m["against_loser"]][:4]
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            "You are the evolution agent of a co-scientist swarm. Improve the hypothesis below. Keep its core idea, fix every "
            "weakness the reviewers and judges raised, correct any data names or cohort numbers that are not supported by the "
            "goal's information (say 'not in the files' rather than invent a number), sharpen the design and controls, and keep "
            "the full format the goal asks for.\n\n"
            f"HYPOTHESIS: {h['title']}\n{h['text']}\n\nREVIEW: {' | '.join(h['review_comments'][-2:])[:1500]}\n"
            f"JUDGES' CRITIQUES: {' | '.join(losses)[:1500]}\nMETA-REVIEW GUIDANCE: {' | '.join(guidance)[:1500]}\n\n"
            + self.data_block(f"{h['title']} {h['text'][:4000]}", min(self.data_budget, 12000))
            + "Return only a JSON object with keys 'title' and 'text'."
        )
        d, reply = self.ask_json("evolution", prompt, 0.6, dict)
        if not d or not d.get("text"):
            self.errors.append(f"evolution of {h['id']}: no parsable rewrite ({reply_snippet(reply)})")
            return None
        return dict(
            id=self.new_id("V"),
            title=as_text(d.get("title") or h["title"]),
            text=as_text(d["text"]),
            worker=h["worker"],
            lens=h["lens"],
            round=h["round"] + 1,
            parent_ids=[h["id"]],
            elo=h["elo"],
            wins=0,
            losses=0,
            review_comments=[],
            references=[],
            active=True,
        )

    # ------------------------------------------------------------------ the run
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        self.progress("orchestrator: splitting the goal into directions")
        lenses = self.orchestrate()
        self.progress(f"{len(lenses)} workers generating in parallel")
        pool = [h for hs in self.pmap(self.work, list(enumerate(lenses))) for h in hs]
        self.progress(f"proximity: de-duplicating {len(pool)} hypotheses")
        self.dedupe(pool)
        if not pool:
            self.errors.append(
                "no hypotheses were generated, so ranking and review were skipped: every worker reply was empty or "
                "unreadable (see above). Try again, or choose a larger model."
            )
            self.progress("stopped: no hypotheses to rank")
        for r in range(1, (self.rounds if pool else 0) + 1):
            todo = [h for h in self.hyps.values() if h["active"] and not h["review_comments"]]
            self.progress(f"round {r}: reflection on {len(todo)} hypotheses")
            self.pmap(self.reflect, todo)
            self.progress(f"round {r}: tournament ({self.matches_per_round} Swiss rounds)")
            self.tournament(r)
            meta = self.meta_review(r)
            if r < self.rounds:
                top = sorted([h for h in self.hyps.values() if h["active"]], key=lambda h: -h["elo"])[
                    : self.top_k_evolve
                ]
                self.progress(f"round {r}: evolving the top {len(top)}")
                new = [x for x in self.pmap(self.evolve, [(h, meta.get("guidance", [])) for h in top]) if x]
                for x in new:
                    x["round"] = r + 1
                self.dedupe(new)
        final = sorted([h for h in self.hyps.values() if h["active"]], key=lambda h: -h["elo"])[: self.final_k]
        if self.data:
            for h in self.hyps.values():
                h["data_check"] = self.data.check_numbers(h["title"] + "\n" + h["text"])
        self.progress(f"done: {len(self.trace)} LLM calls in {time.time() - t0:.0f}s")
        return dict(
            settings=dict(
                model=self.model,
                n_workers=self.n_workers,
                per_worker=self.per_worker,
                rounds=self.rounds,
                matches_per_round=self.matches_per_round,
                top_k_evolve=self.top_k_evolve,
                final_k=self.final_k,
                dedupe_threshold=self.dedupe_threshold,
                parallel=self.parallel,
                k_factor=self.k,
                data=self.data.summary() if self.data else None,
                web=self.web.describe() if self.web else None,
                tools=self.tools.describe() if self.tools else None,
            ),
            lenses=lenses,
            web_search=self.web.log if self.web else [],
            tool_calls=self.tools.log if self.tools else [],
            hypotheses=list(self.hyps.values()),
            final=[h["id"] for h in final],
            merged=self.merged,
            matches=self.matches,
            meta_reviews=self.meta_reviews,
            errors=self.errors,
            seconds=round(time.time() - t0, 1),
            trace=self.trace,
        )


def format_swarm_results(result: Dict[str, Any]) -> str:
    """HTML for the Gradio results panel."""
    import html as H

    from .data_context import data_check_html

    by = {h["id"]: h for h in result["hypotheses"]}
    s = result["settings"]
    out = [
        f"<h2>🐝 Swarm result</h2><p>{s['n_workers']} workers × {s['per_worker']} hypotheses, {s['rounds']} rounds, "
        f"{len(result['matches'])} debates, {len(result['trace'])} LLM calls, {result['seconds']:.0f} s.</p>"
    ]
    used = Counter(t.get("model_used") for t in result["trace"] if t.get("model_used"))
    if used:
        chosen = s.get("model") or ""
        names = ", ".join(f"{H.escape(m)} ({n} call{'s' if n != 1 else ''})" for m, n in used.most_common())
        note = "" if not chosen or set(used) == {chosen} else " (some calls fell back to other models)"
        out.append(f"<p><b>Models that answered:</b> {names}{note}</p>")
    if not result["final"]:
        out.append(
            "<p style='padding:10px;border-radius:6px;background:rgba(231,76,60,0.12)'><b>No hypotheses were generated.</b> "
            "The workers' replies were empty or could not be read, so nothing was ranked or reviewed. The usual cause is a "
            "free model that is busy upstream, or a fallback model too small for the goal and data. Try again in a minute "
            "or choose a larger model.</p>"
        )
    if result["errors"]:
        out.append(
            "<details><summary>⚠️ %d problems</summary><ul>%s</ul></details>"
            % (len(result["errors"]), "".join(f"<li>{H.escape(e)}</li>" for e in result["errors"]))
        )
    out.append(
        "<h3>Directions</h3><ol>"
        + "".join(
            f"<li><b>{H.escape(x.get('name', ''))}</b>: {H.escape(x.get('objective', ''))}</li>"
            for x in result["lenses"]
        )
        + "</ol>"
    )
    if s.get("data"):
        out.append(f"<p><b>Uploaded data:</b> {H.escape(s['data'])}</p>")
    out.append("<h3>Final ranking</h3>")
    for k, hid in enumerate(result["final"], 1):
        h = by[hid]
        out.append(
            f"<details><summary><b>#{k}</b> {H.escape(h['title'])} (Elo {h['elo']:.0f}; {h['wins']}–{h['losses']}; worker {h['worker']}"
            f"{'; evolved from ' + ', '.join(h['parent_ids']) if h['parent_ids'] else ''})</summary>"
            f"<pre style='white-space:pre-wrap;color:inherit;background:rgba(127,127,127,0.10);padding:10px;border-radius:6px'>{H.escape(h['text'])}</pre>"
            + (data_check_html(h["data_check"]) if "data_check" in h else "")
            + f"<p><i>Review:</i> {H.escape(' | '.join(h['review_comments']))}</p></details>"
        )
    for m in result["meta_reviews"]:
        out.append(
            f"<h3>Meta-review, round {m['round']}</h3><ul>"
            + "".join(f"<li>{H.escape(x)}</li>" for x in m.get("critique", []) + m.get("guidance", []))
            + "</ul>"
        )
    return "\n".join(out)
