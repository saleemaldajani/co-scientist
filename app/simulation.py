"""Optional simulation check: the co-scientist tests its top-ranked ideas with simulation tools served over MCP.

After a run has ranked its ideas, each of the top ideas goes through four steps:

1. discover: list the tools of the chosen MCP server (name, description, arguments);
2. design: the model writes a simulation question that would test the idea and up to three tool calls that answer it;
3. run: the calls go to the server (a separate process over stdio, with a time limit per batch);
4. evaluate: the model reads the results, gives a verdict (supported, refuted or inconclusive) with the evidence, and
   rewrites the idea so that every number it states agrees with the tool results.

The checked ideas are listed first by verdict (supported, inconclusive, refuted), keeping their earlier order within
a verdict, and the ideas that were not checked follow unchanged. Two servers ship with the app (app/sim_servers):

- records: computes rates, ratios and held-out tests from the attached tables (and simulates the Broad Street outbreak
  for the London cholera records);
- teams: simulates teams of specialists and generalists completing multi-skill tasks.

Any other stdio MCP server can be added with SimulationLab(command=[...]). Nothing runs unless a server is chosen, so
runs without a simulation check are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SERVERS = {
    "records": dict(
        label="Records lab (computes from the attached tables)", module="app.sim_servers.records_lab", needs_data=True
    ),
    "teams": dict(
        label="Team lab (simulates specialist and generalist teams)",
        module="app.sim_servers.team_lab",
        needs_data=False,
    ),
}
VERDICTS = ("supported", "inconclusive", "refuted")
REPAIR = (
    "\n\nYour previous reply could not be read as a JSON object with at least one call. Reply with only that JSON "
    "object, using the tool names listed above."
)


def server_choices() -> List[str]:
    return ["No simulation"] + [v["label"] for v in SERVERS.values()]


def server_from_choice(choice: str) -> Optional[str]:
    return next((k for k, v in SERVERS.items() if v["label"] == choice), None)


def _json_obj(text: str) -> Dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start != -1:
        depth, inside, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if inside:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    inside = False
                continue
            if c == '"':
                inside = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    return {}


class SimulationLab:
    """An MCP client for one simulation server plus the design, run and evaluate workflow."""

    def __init__(
        self,
        server: str = "records",
        data_dir: Optional[str] = None,
        command: Optional[Sequence[str]] = None,
        top_k: int = 3,
        max_calls: int = 3,
        timeout: float = 120.0,
        result_chars: int = 4000,
    ):
        if command is None:
            if server not in SERVERS:
                raise ValueError(f"unknown simulation server {server!r}; choose from {', '.join(SERVERS)}")
            if SERVERS[server]["needs_data"] and not data_dir:
                raise ValueError(f"the {server} server needs attached data files")
            command = [sys.executable, "-m", SERVERS[server]["module"]] + (
                [os.path.abspath(data_dir)] if data_dir else []
            )
        self.server, self.command = server, list(command)
        self.data_dir = data_dir
        self.top_k, self.max_calls, self.timeout, self.result_chars = top_k, max_calls, timeout, result_chars
        self.tools: List[Dict[str, Any]] = []
        self.log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return True

    def describe(self) -> str:
        return (
            f"simulation check with the {self.server} server: the top {self.top_k} ideas each get a simulation question "
            f"and up to {self.max_calls} tool calls, then a verdict and a revision that agrees with the results"
        )

    # ------------------------------------------------------------------ MCP
    def _params(self):
        from mcp.client.stdio import StdioServerParameters

        env = dict(os.environ)
        env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
        for k in list(env):  # the simulation servers never need the model provider's key
            if k.endswith("_API_KEY"):
                env.pop(k)
        return StdioServerParameters(command=self.command[0], args=self.command[1:], env=env, cwd=REPO)

    async def _session(self, work):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(self._params()) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await work(s)

    def _run(self, work):
        return asyncio.run(asyncio.wait_for(self._session(work), self.timeout))

    def discover(self) -> List[Dict[str, Any]]:
        async def work(s):
            res = await s.list_tools()
            out = []
            for t in res.tools:
                schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                out.append(dict(name=t.name, description=(t.description or "").strip(), schema=schema))
            return out

        self.tools = self._run(work)
        return self.tools

    def call_many(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        names = {t["name"] for t in self.tools}

        async def work(s):
            out = []
            for c in calls:
                name, args = c.get("tool"), c.get("arguments") or {}
                if name not in names or not isinstance(args, dict):
                    out.append(dict(tool=name, arguments=args, ok=False, output=f"unknown tool or arguments: {name!r}"))
                    continue
                try:
                    res = await s.call_tool(name, args)
                    text = "\n".join(getattr(p, "text", "") for p in res.content)
                    bad = (
                        bool(getattr(res, "is_error", False) or getattr(res, "isError", False))
                        or '"error"' in text[:40]
                    )
                    out.append(dict(tool=name, arguments=args, ok=not bad, output=text[: self.result_chars]))
                except Exception as e:  # a failed call is evidence too: report it, keep going
                    out.append(dict(tool=name, arguments=args, ok=False, output=f"{type(e).__name__}: {e}"))
            return out

        try:
            return self._run(work)
        except Exception as e:
            return [
                dict(tool=c.get("tool"), arguments=c.get("arguments"), ok=False, output=f"server failed: {e}")
                for c in calls
            ]

    # ------------------------------------------------------------------ prompts
    def tools_text(self) -> str:
        lines = []
        for t in self.tools:
            props = (t["schema"] or {}).get("properties", {})
            req = set((t["schema"] or {}).get("required", []))
            args = ", ".join(
                f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}"
                + (f" = {json.dumps(v['default'])}" if "default" in v else "")
                for k, v in props.items()
            )
            lines.append(f"- {t['name']}({args}): {t['description']}")
        return "\n".join(lines)

    def design_prompt(self, goal: str, idea: Dict[str, Any], data: str = "") -> str:
        return (
            f"Research goal: {goal}\n\n"
            + (f"Attached data (the tools read these files):\n{data}\n\n" if data else "")
            + f"A hypothesis ranked near the top:\nTITLE: {idea.get('title', '')}\n{idea.get('text', '')}\n\n"
            "You can check it with these simulation and analysis tools before it is reported:\n"
            + self.tools_text()
            + "\n\nWrite one simulation question whose answer would support or refute this hypothesis, and up to "
            f"{self.max_calls} tool calls that answer it. Prefer calls that recompute the numbers the hypothesis states "
            "or that test its central prediction. Return only JSON: "
            '{"question": "...", "calls": [{"tool": "<tool name>", "arguments": {"<name>": <value>}}]}'
        )

    def evaluate_prompt(self, goal: str, idea: Dict[str, Any], question: str, results: List[Dict[str, Any]]) -> str:
        res = "\n\n".join(
            f"### {r['tool']}({json.dumps(r['arguments'])}){'' if r['ok'] else ' FAILED'}\n{r['output']}"
            for r in results
        )
        return (
            f"Research goal: {goal}\n\nHypothesis:\nTITLE: {idea.get('title', '')}\n{idea.get('text', '')}\n\n"
            f"Simulation question: {question}\n\nTool results (computed from the data, not by a language model):\n"
            f"{res or '(no call ran)'}\n\n"
            "Judge the hypothesis against these results. Then rewrite it so that every number it states agrees with "
            "the tool results, quoting the tool for each number you take from them; keep its reasoning and predictions, "
            "and say plainly where the results contradict it. Return only JSON: "
            '{"verdict": "supported" | "inconclusive" | "refuted", "evidence": "<two or three sentences citing the '
            'numbers>", "revised_title": "...", "revised_text": "..."}'
        )

    # ------------------------------------------------------------------ workflow
    def check(
        self,
        ask: Callable[[str], str],
        goal: str,
        ideas: List[Dict[str, Any]],
        data: str = "",
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Check the top ideas; return {server, tools, log, final} with final the re-ordered, revised idea list."""
        say = progress or (lambda m: logger.info("[simulation] %s", m))
        try:
            self.discover()
        except Exception as e:
            say(f"simulation server unavailable: {e}")
            return dict(server=self.server, tools=[], log=[], final=list(ideas), error=str(e))
        say(f"simulation server: {len(self.tools)} tools")
        checked = []
        for k, idea in enumerate(ideas[: self.top_k], 1):
            say(f"simulation check {k}/{min(self.top_k, len(ideas))}: designing")
            prompt = self.design_prompt(goal, idea, data)
            design_reply = ask(prompt) or ""
            plan = _json_obj(design_reply)
            if not plan.get("calls"):  # unreadable or empty plan: ask once more for the JSON alone
                design_reply += "\n\n[retry]\n" + (ask(prompt + REPAIR) or "")
                plan = _json_obj(design_reply.rsplit("[retry]", 1)[-1]) or plan
            calls = [c for c in plan.get("calls") or [] if isinstance(c, dict)][: self.max_calls]
            question = str(plan.get("question") or "")
            say(f"simulation check {k}: running {len(calls)} calls")
            results = self.call_many(calls) if calls else []
            if any(r["ok"] for r in results):
                eval_reply = ask(self.evaluate_prompt(goal, idea, question, results)) or ""
                ev = _json_obj(eval_reply)
            else:  # nothing was computed, so there is nothing to judge the idea by: leave it as it was
                eval_reply = ""
                ev = dict(
                    verdict="inconclusive", evidence="No simulation call ran successfully, so the idea is unchanged."
                )
            verdict = str(ev.get("verdict", "")).strip().lower()
            verdict = verdict if verdict in VERDICTS else "inconclusive"
            evidence = str(ev.get("evidence") or "").strip()
            text = str(ev.get("revised_text") or "").strip() or idea.get("text", "")
            title = str(ev.get("revised_title") or "").strip() or idea.get("title", "")
            entry = dict(idea)
            entry.update(
                title=title,
                text=text + f"\n\nSimulation check ({self.server} lab): {verdict}. {evidence}".rstrip(),
                verdict=verdict,
                original_title=idea.get("title", ""),
                original_text=idea.get("text", ""),
            )
            rec = dict(
                idea_id=idea.get("id"),
                question=question,
                calls=calls,
                results=results,
                verdict=verdict,
                evidence=evidence,
                design_reply=design_reply,
                evaluate_reply=eval_reply,
                calls_ok=sum(r["ok"] for r in results),
            )
            with self._lock:
                self.log.append(rec)
            checked.append(entry)
        order = {v: i for i, v in enumerate(VERDICTS)}
        ranked = [e for _, e in sorted(enumerate(checked), key=lambda x: (order[x[1]["verdict"]], x[0]))]
        return dict(server=self.server, tools=self.tools, log=self.log, final=ranked + list(ideas[self.top_k :]))


def format_simulation_html(sim: Optional[Dict[str, Any]]) -> str:
    """HTML section for the results panel."""
    import html as H

    if not sim or not sim.get("log"):
        return ""
    out = [f"<h3>🔬 Simulation check ({H.escape(sim['server'])} lab, {len(sim['tools'])} tools)</h3>"]
    for r in sim["log"]:
        calls = "".join(
            f"<li><code>{H.escape(x['tool'] or '')}({H.escape(json.dumps(x['arguments']))})</code>"
            f"{'' if x['ok'] else ' ⚠️ failed'}</li>"
            for x in r["results"]
        )
        out.append(
            f"<details><summary><b>{H.escape(r['verdict'])}</b>: {H.escape(r['question'] or '(no question)')}</summary>"
            f"<ul>{calls or '<li>no call ran</li>'}</ul><p>{H.escape(r['evidence'])}</p></details>"
        )
    return "\n".join(out)
