"""Export results: a run's ranked ideas as Markdown, HTML and JSON.

Layout:
    <out>/share/ideas_bundle.json   one object per idea (with the direction of the worker that wrote it, or null)
    <out>/share/directions.json     swarm runs: one record per assigned worker, in assignment order ([] otherwise)
    <out>/share/tool_calls.json     runs with science tools: every step's chosen tools, calls and output ([] otherwise)
    <out>/share/ideas_bundle.md     directions, leaderboard, then every idea (summary, full idea, review, references)
    <out>/share/ideas_bundle.html   the same as one HTML page
    <out>/share/idea_rankNN.html    one page per idea, with navigation between ranks
    <out>/reviews/review_<id>.md    the review of each idea
    <out>/references.csv, .md       every reference, with the idea it is attached to
An "idea" is one ranked hypothesis. The reviewer gives HIGH / MEDIUM / LOW labels, which are reported as labels (the numeric
score fields are null), and swarm mode adds each idea's debate record.
"""

import csv
import html
import json
import os
import re
import zipfile
from typing import Any, Dict, List, Optional

try:
    from markdown_it import MarkdownIt

    _MD = MarkdownIt("commonmark", {"html": False}).enable("table")
except Exception:  # pragma: no cover - fallback keeps the export working without the renderer
    _MD = None

CSS = """<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.55;color:#202124;
      max-width:980px;margin:32px auto;padding:0 24px;background:#fff}
 h1{font-size:1.6em;border-bottom:2px solid #e8eaed;padding-bottom:8px}
 h2{font-size:1.25em;margin-top:1.8em;color:#174ea6} h3{font-size:1.08em;margin-top:1.4em}
 h4{margin-top:1.2em}
 .meta{color:#5f6368;font-size:0.92em;margin:-4px 0 18px}
 .meta b{color:#202124}
 .nav{font-size:0.9em;margin-bottom:14px} .nav a{margin-right:14px}
 table{border-collapse:collapse;margin:10px 0;font-size:0.93em}
 th,td{border:1px solid #dadce0;padding:5px 9px;vertical-align:top;text-align:left}
 th{background:#f1f3f4}
 code{background:#f1f3f4;padding:1px 4px;border-radius:3px;font-size:0.92em}
 pre{background:#f8f9fa;padding:10px;border-radius:6px;overflow-x:auto}
 details{margin:18px 0;border:1px solid #dadce0;border-radius:6px;padding:8px 14px;background:#fafafa}
 summary{cursor:pointer;font-weight:600;color:#174ea6}
 blockquote{border-left:4px solid #dadce0;margin:0;padding:2px 14px;color:#5f6368}
 hr{border:0;border-top:1px solid #e8eaed;margin:28px 0}
 .badge{display:inline-block;background:#e8f0fe;color:#174ea6;border-radius:12px;padding:1px 10px;margin-right:6px;font-size:0.85em}
 .direction{border-left:4px solid #174ea6;background:#f8f9fa;padding:6px 12px;margin:8px 0 14px;white-space:pre-wrap}
 @media (prefers-color-scheme: dark){
  body{color:#e8eaed;background:#121417} h1{border-bottom-color:#3c4043} h2,summary{color:#8ab4f8}
  .meta{color:#9aa0a6} .meta b{color:#e8eaed} a{color:#8ab4f8}
  th,td{border-color:#3c4043} th{background:#202124} code{background:#2a2c30} pre{background:#1b1d21}
  details{background:#1b1d21;border-color:#3c4043} blockquote{border-left-color:#3c4043;color:#9aa0a6} hr{border-top-color:#3c4043}
  .badge{background:#1f3a5f;color:#d2e3fc} .direction{background:#1b1d21;border-left-color:#8ab4f8}
 }
</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)'],['$','$']],displayMath:[['\\\\[','\\\\]'],['$$','$$']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""


def md_to_html(text: str) -> str:
    if _MD is None:
        return f"<pre>{html.escape(text)}</pre>"
    return _MD.render(text or "")


def _as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return "\n\n".join(f"**{k}**: {_as_text(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "\n".join(_as_text(x) for x in v)
    return "" if v is None else str(v)


def summary_of(text: str, n: int = 900) -> str:
    """The hypothesis sentence if the text labels one, else the opening of the text."""
    t = _as_text(text)
    m = re.search(r"\*\*\s*(?:\d+\s*[.)]\s*)?Hypothesis\s*\*\*\s*:?\s*(.+?)(?:\n\s*\n|\n\s*\*\*)", t, re.S | re.I)
    s = (m.group(1) if m else t).strip()
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def _ref_row(ref: Any) -> Dict[str, str]:
    t = _as_text(ref).strip()
    doi = re.search(r"10\.\d{4,9}/[^\s,;\"'<>\])]+", t)
    arx = re.search(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?", t)
    link = (
        f"https://doi.org/{doi.group(0).rstrip('.')}"
        if doi
        else (f"https://arxiv.org/abs/{arx.group(1)}" if arx else "")
    )
    return dict(
        title=t,
        authors="",
        date="",
        journal="",
        source="model-written reference (not retrieved)",
        doi=doi.group(0).rstrip(".") if doi else "",
        link=link,
    )


def ideas_from_hypotheses(
    hyps: List[Dict[str, Any]], matches: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Turn hypothesis dicts (single-cycle Hypothesis.to_dict() or swarm hypotheses) into ranked ideas."""
    by_id = {h.get("id"): h for h in hyps}
    ranked = sorted(hyps, key=lambda h: -float(h.get("elo_score", h.get("elo", 1200)) or 0))
    ideas = []
    for k, h in enumerate(ranked, 1):
        hid = h.get("id")
        text = _as_text(h.get("text"))
        review = []
        if h.get("novelty_review") or h.get("feasibility_review"):
            review.append(
                f"**Novelty:** {h.get('novelty_review') or '-'} · **Feasibility:** {h.get('feasibility_review') or '-'}"
            )
        for c in h.get("review_comments") or []:
            review.append(_as_text(c))
        if h.get("parent_ids"):
            review.append(
                "**Evolved from:** "
                + ", ".join(f"{p} ({_as_text(by_id.get(p, {}).get('title', ''))})" for p in h["parent_ids"])
            )
        dc = h.get("data_check")
        if dc is not None:
            found = sum(1 for c in dc if c.get("found"))
            review.append(
                f"## Data check ({found} of {len(dc)} quoted numbers found in the uploaded data)"
                if dc
                else "## Data check (no numbers quoted)"
            )
            for c in dc:
                review.append(
                    f"- {'found' if c.get('found') else 'NOT FOUND'}: {c.get('number')}"
                    + (f" ({'; '.join(c.get('where') or [])})" if c.get("found") else "")
                )
        mm = [m for m in matches or [] if hid in (m.get("winner"), m.get("loser"))]
        if mm:
            review.append(
                f"## Tournament ({sum(m['winner'] == hid for m in mm)} wins, {sum(m['loser'] == hid for m in mm)} losses)"
            )
            for m in mm:
                other = m["loser"] if m["winner"] == hid else m["winner"]
                res = "won against" if m["winner"] == hid else "lost to"
                review.append(
                    f"- Round {m.get('round')}, debate {m.get('match')}: {res} {other} ({_as_text(by_id.get(other, {}).get('title', ''))[:90]}). "
                    f"Judge: {_as_text(m.get('reason'))}"
                )
        ideas.append(
            dict(
                rank=k,
                id=hid,
                title=_as_text(h.get("title")),
                summary=summary_of(text),
                text=text,
                elo_rating=float(h.get("elo_score", h.get("elo", 1200)) or 0),
                review="\n\n".join(review),
                references=[_ref_row(r) for r in h.get("references") or []],
                review_score=None,
                correctness_score=None,
                novelty_score=None,
                feasibility_score=None,
                novelty_review=h.get("novelty_review"),
                feasibility_review=h.get("feasibility_review"),
                wins=h.get("wins"),
                losses=h.get("losses"),
                worker=h.get("worker"),
                lens=h.get("lens"),
                parent_ids=h.get("parent_ids") or [],
                tools_used="",
            )
        )
    return ideas


def directions_from_lenses(lenses: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """A swarm run's stored assignments as export records, in assignment order.

    The live results page lists the same records (``result["lenses"]``): worker k + 1 got ``lenses[k]``, and its
    hypotheses carry ``worker = k + 1``. Text is copied as stored; nothing is regenerated or inferred.
    """
    return [
        {"worker": k, "lens": _as_text(lens.get("name", "")), "direction": _as_text(lens.get("objective", ""))}
        for k, lens in enumerate(lenses or [], 1)
        if isinstance(lens, dict)
    ]


def _direction_html(i: Dict[str, Any]) -> str:
    if not i.get("direction"):
        return ""
    who = f"Worker {i.get('worker')}" + (f" · {i['lens']}" if i.get("lens") else "")
    return f'<p class="direction"><b>Direction</b> ({html.escape(who)}): {html.escape(i["direction"])}</p>'


def _badges(i: Dict[str, Any]) -> str:
    b = [f"Elo {i['elo_rating']:.1f}"]
    if i.get("novelty_review"):
        b.append(f"novelty {i['novelty_review']}")
    if i.get("feasibility_review"):
        b.append(f"feasibility {i['feasibility_review']}")
    if i.get("wins") is not None and i.get("losses") is not None:
        b.append(f"debates {i['wins']}–{i['losses']}")
    if i.get("lens"):
        b.append(f"worker {i.get('worker')}: {i['lens']}")
    return b


def write_bundle(
    ideas: List[Dict[str, Any]],
    out_dir: str,
    run_name: str,
    goal: str,
    mode: str = "single",
    model: str = "",
    input_note: str = "the research-goal text only (no files)",
    directions: Optional[List[Dict[str, Any]]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Write the bundle to out_dir and a zip next to it; return the zip path.

    ``directions``: the run's stored worker assignments (``directions_from_lenses(result["lenses"])`` for a swarm run;
    None for single-mode or older runs). Each idea is joined to its assignment by worker ID only; an idea without one
    gets ``direction: null`` and no direction text on the rendered pages.
    """
    share, revdir = os.path.join(out_dir, "share"), os.path.join(out_dir, "reviews")
    os.makedirs(share, exist_ok=True)
    os.makedirs(revdir, exist_ok=True)
    directions = [dict(worker=d["worker"], lens=d["lens"], direction=d["direction"]) for d in directions or []]
    by_worker = {str(d["worker"]): d["direction"] for d in directions}
    for i in ideas:
        i.update(
            goal=goal,
            run_dir=run_name,
            idea_md_path=f"share/ideas_bundle.md#rank-{i['rank']}",
            review_md_path=f"reviews/review_{i['id']}.md",
            direction=by_worker.get(str(i.get("worker"))) if i.get("worker") is not None else None,
        )
    json.dump(ideas, open(os.path.join(share, "ideas_bundle.json"), "w"), indent=1, ensure_ascii=False, default=str)
    json.dump(directions, open(os.path.join(share, "directions.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(
        tool_calls or [], open(os.path.join(share, "tool_calls.json"), "w"), indent=1, ensure_ascii=False, default=str
    )

    lb = [
        "| rank | id | title | elo_rating | novelty | feasibility | debates | tools_used |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lb += [
        f"| {i['rank']} | {i['id']} | {i['title'].replace('|', '/')} | {i['elo_rating']:.1f} | {i.get('novelty_review') or ''} | "
        f"{i.get('feasibility_review') or ''} | {'' if i.get('wins') is None else str(i['wins']) + '–' + str(i['losses'])} |  |"
        for i in ideas
    ]
    head = (
        f"# Co-Scientist ideas - {run_name}\n\nMode: {mode}{' · model: ' + model if model else ''}\n\n"
        f"Input data: {input_note}\n\n"
        + (
            "## Directions\n\n"
            + "\n".join(f"- **Worker {d['worker']} · {d['lens']}**: {d['direction']}" for d in directions)
            + "\n\n"
            if directions
            else ""
        )
        + "## Leaderboard\n\n"
        + "\n".join(lb)
        + "\n\n"
    )
    blocks = []
    for i in ideas:
        refs = (
            "\n".join(
                f"{k}. {r['title']}" + (f" ({r['link']})" if r["link"] else "")
                for k, r in enumerate(i["references"], 1)
            )
            or "None attached."
        )
        blocks.append(
            f"---\n\n# Rank {i['rank']}: {i['title']}\n\n*Idea ID* `{i['id']}` · "
            + " · ".join(_badges(i))
            + (
                f"\n\n**Direction** (Worker {i.get('worker')}{' · ' + i['lens'] if i.get('lens') else ''}): {i['direction']}"
                if i.get("direction")
                else ""
            )
            + f"\n\n## Summary\n\n{i['summary']}\n\n## Full idea\n\n{i['text']}\n\n## Review\n\n{i['review'] or 'No review recorded.'}\n\n## References\n\n{refs}\n"
        )
        open(os.path.join(revdir, f"review_{i['id']}.md"), "w").write(
            f"# Review of idea {i['id']}: {i['title']}\n\n{i['review'] or 'No review recorded.'}\n"
        )
    md = head + "\n".join(blocks) + f"\n---\n\n## Goal\n\n{goal}\n"
    open(os.path.join(share, "ideas_bundle.md"), "w").write(md)

    nav = (
        '<div class="nav"><a href="ideas_bundle.html">&larr; all ideas</a>'
        + "".join(f'<a href="idea_rank{i["rank"]:02d}.html">#{i["rank"]}</a>' for i in ideas)
        + "</div>"
    )

    def page(title: str, body: str) -> str:
        return f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>\n{CSS}\n</head><body>{body}</body></html>'

    def idea_html(i: Dict[str, Any]) -> str:
        badges = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in _badges(i))
        refs = "".join(
            f"<li>{html.escape(r['title'])}"
            + (f' <a href="{html.escape(r["link"])}">link</a>' if r["link"] else "")
            + "</li>"
            for r in i["references"]
        )
        return (
            f'<h1>Rank {i["rank"]}: {html.escape(i["title"])}</h1><p class="meta">{badges}<br>Idea ID <code>{html.escape(str(i["id"]))}</code> &middot; run <b>{html.escape(run_name)}</b></p>'
            + _direction_html(i)
            + f"<h2>Summary</h2>{md_to_html(i['summary'])}<h2>Full idea</h2>{md_to_html(i['text'])}"
            f"<details><summary>Review</summary>{md_to_html(i['review'] or 'No review recorded.')}</details>"
            f"<h2>References</h2>{'<ol>' + refs + '</ol>' if refs else '<p>None attached.</p>'}"
        )

    for i in ideas:
        open(os.path.join(share, f"idea_rank{i['rank']:02d}.html"), "w").write(
            page(f"Rank {i['rank']}: {i['title']}", nav + idea_html(i))
        )
    rows = "".join(
        f'<tr><td>{i["rank"]}</td><td><a href="idea_rank{i["rank"]:02d}.html">{html.escape(i["title"])}</a></td><td>{i["elo_rating"]:.1f}</td>'
        f"<td>{html.escape(str(i.get('novelty_review') or ''))}</td><td>{html.escape(str(i.get('feasibility_review') or ''))}</td></tr>"
        for i in ideas
    )
    bundle = (
        f'<h1>Co-Scientist ideas &ndash; {html.escape(run_name)}</h1><p class="meta">Mode <b>{html.escape(mode)}</b>'
        f"{' &middot; model <b>' + html.escape(model) + '</b>' if model else ''} &middot; input: {html.escape(input_note)}</p>"
        + (
            "<h2>Directions</h2><table><tr><th>worker</th><th>lens</th><th>direction</th></tr>"
            + "".join(
                f"<tr><td>{html.escape(str(d['worker']))}</td><td>{html.escape(d['lens'])}</td><td>{html.escape(d['direction'])}</td></tr>"
                for d in directions
            )
            + "</table>"
            if directions
            else ""
        )
        + f"<h2>Leaderboard</h2><table><tr><th>rank</th><th>title</th><th>Elo</th><th>novelty</th><th>feasibility</th></tr>{rows}</table>"
        + "".join("<hr>" + idea_html(i) for i in ideas)
        + f"<hr><h2>Goal</h2>{md_to_html(goal)}"
    )
    open(os.path.join(share, "ideas_bundle.html"), "w").write(page(f"Co-Scientist ideas – {run_name}", bundle))

    fields = ["idea_rank", "idea_id", "roles", "title", "authors", "date", "journal", "source", "doi", "link"]
    with open(os.path.join(out_dir, "references.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in ideas:
            for r in i["references"]:
                w.writerow(dict(r, idea_rank=i["rank"], idea_id=i["id"], roles="cited in review"))
    n = sum(len(i["references"]) for i in ideas)
    open(os.path.join(out_dir, "references.md"), "w").write(
        f"# References - {run_name}\n\n{n} references attached by the reflection agent. They are written by the model, not retrieved: "
        "resolve each before relying on it.\n\n"
        + "\n".join(f"- Rank {i['rank']} ({i['id']}): {r['title']}" for i in ideas for r in i["references"])
        + "\n"
    )

    zpath = out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.join(os.path.basename(out_dir.rstrip("/")), os.path.relpath(p, out_dir)))
    return zpath
