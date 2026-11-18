"""Optional science tools before each generation step, for single and swarm mode alike.

The tools are the scripts in app/science_skills/ (databases such as UniProt, PubMed, ChEMBL, Ensembl, ClinVar and
ClinicalTrials.gov, and literature search). When a run has tools on, every step that writes new hypotheses (a
single-mode generation call; each swarm worker) first:

1. asks the model which of the enabled tools would help (skipped when `max_skills` or fewer are enabled),
2. shows the model those tools' command lines (each script's --help text and the examples from its SKILL.md) and asks
   for up to `max_calls` calls as JSON: {"tool": ..., "script": ..., "args": [...]},
3. runs each call as `python <script> <args>` (no shell, a time limit, a scratch folder, only scripts of enabled
   tools),
4. shows the model that output and asks for up to `max_calls` follow-up calls (for example, the abstracts of the
   articles a search found), runs them the same way (`rounds` = 2; 1 turns follow-ups off; skipped if no call ran),
5. puts the output of both rounds into that step's prompt as a TOOL RESULTS block, cut to a character budget.

Nothing else changes, so a run with tools off sends exactly the prompts it sent before. Every request, call and
output is kept in `ScienceTools.log`, so a reader can check where a claim came from.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "science_skills")
PREAMBLE = (
    "TOOL RESULTS - the output of the scientific database and tool calls you asked for, run {date}. They can be "
    "incomplete or wrong. Name the tool and query for anything you take from them, and keep numbers from these tools "
    "apart from numbers in any attached data."
)


# Display names for the checkbox list (the folder name is what the model sees and the log records).
LABELS = {
    "alphafold_database_fetch_and_analyze": "AlphaFold DB structures",
    "chembl_database": "ChEMBL compounds and bioactivity",
    "clinical_trials_database": "ClinicalTrials.gov",
    "clinvar_database": "ClinVar variants",
    "dbsnp_database": "dbSNP variants",
    "embl_ebi_ols": "EMBL-EBI ontologies (OLS)",
    "encode_ccres_database": "ENCODE regulatory elements",
    "ensembl_database": "Ensembl genes",
    "foldseek_structural_search": "Foldseek structure search",
    "gnomad_database": "gnomAD population variants",
    "gtex_database": "GTEx tissue expression",
    "human_protein_atlas_database": "Human Protein Atlas",
    "interpro_database": "InterPro protein families",
    "jaspar_database": "JASPAR binding motifs",
    "literature_search_arxiv": "arXiv papers",
    "literature_search_biorxiv": "bioRxiv preprints",
    "literature_search_europepmc": "Europe PMC literature",
    "literature_search_openalex": "OpenAlex scholarly works",
    "ncbi_sequence_fetch": "NCBI sequences",
    "openfda_database": "openFDA drugs and adverse events",
    "opentargets_database": "Open Targets",
    "pdb_database": "PDB structures",
    "protein_sequence_msa": "Protein sequence alignment",
    "protein_sequence_similarity_search": "Protein similarity search",
    "pubchem_database": "PubChem compounds",
    "pubmed_database": "PubMed literature",
    "quickgo_database": "QuickGO gene ontology",
    "reactome_database": "Reactome pathways",
    "string_database": "STRING protein interactions",
    "ucsc_conservation_and_tfbs": "UCSC conservation and binding sites",
    "unibind_database": "UniBind binding sites",
    "uniprot_database": "UniProt proteins",
}


def load_catalog(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The skills in catalog.json, by name ({} if the folder is missing)."""
    path = path or os.path.join(SKILLS_DIR, "catalog.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {s["name"]: s for s in json.load(f)["skills"]}


def tool_choices(catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> List[tuple]:
    """(label, name) pairs for a checkbox list, by label, e.g. ('UniProt proteins', 'uniprot_database')."""
    catalog = load_catalog() if catalog is None else catalog
    pairs = [(LABELS.get(n) or s["title"][:1].upper() + s["title"][1:], n) for n, s in catalog.items()]
    return sorted(pairs, key=lambda p: p[0].lower())


def _json_array(reply: str) -> List[Any]:
    m = re.search(r"\[.*\]", reply or "", re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except ValueError:
        return []
    return out if isinstance(out, list) else []


def parse_choice(reply: str, allowed: Sequence[str], n: int) -> List[str]:
    """Tool names the model chose, in its order, limited to the enabled tools."""
    out = []
    names = {a.lower().replace("-", "_"): a for a in allowed}
    for x in _json_array(reply):
        key = str(x).strip().lower().replace("-", "_").replace(" ", "_")
        if key in names and names[key] not in out:
            out.append(names[key])
    return out[:n]


def check_call(call: Any, catalog: Dict[str, Dict[str, Any]], enabled: Sequence[str]) -> Optional[str]:
    """Why a requested call cannot run (None if it can)."""
    if not isinstance(call, dict):
        return "not an object"
    tool, script, args = call.get("tool"), call.get("script"), call.get("args", [])
    if tool not in enabled or tool not in catalog:
        return f"tool {tool!r} is not enabled"
    if script not in catalog[tool]["scripts"]:
        return f"{tool} has no script {script!r}"
    if not isinstance(args, list) or len(args) > 40 or not all(isinstance(a, (str, int, float)) for a in args):
        return "args must be a list of strings"
    for a in map(local_path, map(str, args)):
        if len(a) > 1000 or a.startswith(("/", "~")) or ".." in a.split("/") or "\x00" in a:
            return f"argument not allowed: {a[:60]!r}"
    return None


def local_path(arg: str) -> str:
    """A /tmp/ path (the skills' examples write there) becomes the same name in the call's scratch folder."""
    return arg[len("/tmp/") :] if arg.startswith("/tmp/") and len(arg) > len("/tmp/") else arg


def written_files(folder: str, limit: int = 2_000_000) -> str:
    """The files a call wrote (scripts often write results to --output rather than print them), as text."""
    parts = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel, size = os.path.relpath(p, folder), os.path.getsize(p)
            try:
                if size > limit:
                    raise UnicodeDecodeError("", b"", 0, 1, "large")
                with open(p, encoding="utf-8") as fh:
                    text = fh.read()
                try:  # JSON without its indentation: the same content in far fewer characters of the prompt
                    text = json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
                except ValueError:
                    pass
                parts.append(f"[file {rel}]\n{text}")
            except UnicodeDecodeError:
                parts.append(f"[file {rel}: {size:,} bytes, not shown]")
    return "\n".join(parts)


def fair_shares(sizes: List[int], budget: int) -> List[int]:
    """Characters each output may keep: all of a short one, and an equal share of what is left for the long ones."""
    caps, left, todo = [0] * len(sizes), budget, sorted(range(len(sizes)), key=lambda i: sizes[i])
    for k, i in enumerate(todo):
        caps[i] = min(sizes[i], left // (len(todo) - k))
        left -= caps[i]
    return caps


def default_runner(script_path: str, args: List[str], timeout: float) -> Dict[str, Any]:
    """Run one script in a fresh scratch folder and return its exit code and output."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(script_path) + os.pathsep + env.get("PYTHONPATH", "")
    # the model's own credentials are never passed to a tool
    for k in [
        k
        for k in env
        if k.endswith(("_API_KEY", "_TOKEN")) and k not in ("NCBI_API_KEY", "OPENALEX_API_KEY", "FDA_API_KEY")
    ]:
        env.pop(k)
    with tempfile.TemporaryDirectory(prefix="tool_") as cwd:
        try:
            r = subprocess.run(
                [sys.executable, script_path, *args],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            out = (r.stdout or "") + (("\n" + r.stderr) if r.returncode else "")
            files = written_files(cwd)
            return dict(returncode=r.returncode, output=out + ("\n" + files if files else ""))
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return dict(returncode=None, output=out + f"\n[stopped after {timeout:.0f} s]")


DATA_NOTE = (
    "Attached data: the user attached the files below. You will see their contents in the next step, where you "
    "propose hypotheses; the tools cannot read them. Choose tools and calls that add to these files, for example "
    "published analyses or records they do not contain.\n{outline}"
)
# provider notices addressed to AI agents (one asked agents to send their user's email for an API key): never shown
_AGENT_NOTICE = re.compile(r'\s*"notice"\s*:\s*"(?:[^"\\]|\\.)*"\s*,?')

REQUIRE_CHOICE = "\n\nYou must use at least one tool for this step. Return a JSON array of 1 to {n} tool names."
REQUIRE_CALL = (
    "\n\nYou must run at least one call for this step. Return a JSON array of 1 to 3 call objects, using the scripts "
    "and arguments shown above."
)


class ScienceTools:
    def __init__(
        self,
        enabled: Optional[Sequence[str]] = None,
        max_skills: int = 2,
        max_calls: int = 3,
        rounds: int = 2,
        timeout: float = 60.0,
        output_chars: int = 6000,
        budget: int = 12000,
        usage_chars: int = 8000,
        catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        runner: Optional[Callable[[str, List[str], float], Dict[str, Any]]] = None,
        skills_dir: str = SKILLS_DIR,
        require: bool = False,
    ):
        """enabled: tool names (folder names in app/science_skills/skills); None or empty = every tool.

        require: the model must use at least one tool. It is asked the usual question first, and only when it declines
        (no tool chosen, or no runnable call written) is it asked once more with "at least one"; the log keeps the
        declined reply, so a run with require=True still shows whether the model would have used the tools by itself.
        """
        self.catalog = load_catalog() if catalog is None else catalog
        unknown = [e for e in (enabled or []) if e not in self.catalog]
        if unknown:
            raise ValueError(f"unknown science tools: {unknown}")
        self.enabled = list(enabled) if enabled else sorted(self.catalog)
        self.max_skills, self.max_calls, self.rounds, self.timeout = max_skills, max_calls, max(1, rounds), timeout
        self.output_chars, self.budget, self.usage_chars = output_chars, budget, usage_chars
        self.runner = runner or default_runner
        self.skills_dir = skills_dir
        self.require = require
        self.log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return bool(self.enabled)

    def describe(self) -> str:
        n = len(self.enabled)
        which = "all" if n == len(self.catalog) else ", ".join(self.enabled)
        return (
            f"science tools before each generation step ({n} enabled: {which}): the model picks up to {self.max_skills} "
            f"tools and writes up to {self.max_calls} calls per round in {self.rounds} round(s) (later rounds follow up on "
            f"earlier output), each call limited to {self.timeout:.0f} s; about {self.budget:,} characters of output go "
            "into the prompt" + ("; at least one tool call is required" if self.require else "")
        )

    # ------------------------------------------------------------------ prompts
    def _head(self, goal: str, direction: str, data: str = "") -> str:
        return (
            f"Research Goal: {goal}\n\n"
            + (f"Your direction: {direction}\n\n" if direction else "")
            + (DATA_NOTE.format(outline=data) + "\n\n" if data else "")
        )

    def choose_prompt(self, goal: str, direction: str = "", data: str = "") -> str:
        lines = [f"- {n}: {self.catalog[n]['description']}" for n in self.enabled]
        return (
            self._head(goal, direction, data)
            + "Before you propose hypotheses you can query scientific databases and tools. The tools are:\n"
            + "\n".join(lines)
            + f"\n\nChoose up to {self.max_skills} tools whose results would most help you propose and test hypotheses "
            + "for this goal"
            + (" in your direction" if direction else "")
            + ". Return only a JSON array of tool names, or [] if no tool would help."
        )

    def usage(self, name: str) -> str:
        s = self.catalog[name]
        parts = [f"## Tool: {name}\n{s['description']}"]
        if s.get("examples"):  # first: short and concrete, and the help text below may be cut
            parts.append("Examples (script and arguments):\n" + "\n".join(s["examples"]))
        for script, help_text in s["scripts"].items():
            parts.append(f"### Script: {script}\n{help_text}")
        text = "\n\n".join(parts)
        return text if len(text) <= self.usage_chars else text[: self.usage_chars] + "\n[... usage cut ...]"

    def calls_prompt(self, goal: str, chosen: Sequence[str], direction: str = "", data: str = "") -> str:
        return (
            self._head(goal, direction, data)
            + "Before you propose hypotheses you can run these scientific tools. Each call runs one script with "
            "command-line arguments, with no shell (no pipes, redirects or files from earlier calls), and returns what "
            "it prints and the contents of any files it writes (give output files a plain name, such as result.json)."
            "\n\n"
            + "\n\n".join(self.usage(n) for n in chosen)
            + f"\n\nWrite up to {self.max_calls} calls whose output would help you propose and test hypotheses for this "
            "goal. Use specific identifiers, genes, compounds or search terms, and small result limits. Return only a "
            'JSON array of objects like {"tool": "<tool name>", "script": "<script file>", "args": ["<arg>", ...]}, '
            "or [] if no call would help."
        )

    def followup_prompt(
        self, goal: str, chosen: Sequence[str], done: List[Dict[str, Any]], direction: str = "", data: str = ""
    ) -> str:
        return (
            self._head(goal, direction, data)
            + "You ran these tool calls before proposing hypotheses. Their output so far:\n\n"
            + self.block(done, budget=self.budget // 2, preamble=False)
            + "\n\nThe tools again, for follow-up calls:\n\n"
            + "\n\n".join(self.usage(n) for n in chosen)
            + f"\n\nWrite up to {self.max_calls} follow-up calls that get the details this output points to, for "
            "example the abstracts or records of the three to five most relevant identifiers it returned (articles, "
            "genes, proteins, compounds, trials); each call's output is cut to about "
            + f"{self.output_chars:,} characters. Do not repeat a call above. Return only a JSON array of objects like "
            '{"tool": "<tool name>", "script": "<script file>", "args": ["<arg>", ...]}, or [] if you have what you need.'
        )

    # ------------------------------------------------------------------ running
    def run_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        tool, script, args = call["tool"], call["script"], [local_path(str(a)) for a in call.get("args", [])]
        path = os.path.join(self.skills_dir, "skills", tool, "scripts", script)
        res = self.runner(path, args, self.timeout)
        out = _AGENT_NOTICE.sub("", res.get("output") or "").strip()
        if len(out) > self.output_chars:
            out = out[: self.output_chars] + f"\n[... {len(out) - self.output_chars:,} more characters cut ...]"
        return dict(tool=tool, script=script, args=args, returncode=res.get("returncode"), output=out)

    def block(self, results: List[Dict[str, Any]], budget: Optional[int] = None, preamble: bool = True) -> str:
        """Every call's output; when they do not all fit, short outputs stay whole and the long ones share the rest
        equally (so a follow-up's abstracts are not crowded out by the search that found them)."""
        budget = budget or self.budget
        parts = []
        for r in results:
            cmd = " ".join([r["script"], *[json.dumps(a) if " " in a else a for a in r["args"]]])
            status = "" if r["returncode"] == 0 else f" (exit code {r['returncode']})"
            label = f"round {r['round']} · " if r.get("round", 1) > 1 else ""
            parts.append(f"### {label}{r['tool']}: {cmd}{status}\n{r['output'] or '(no output)'}")
        caps = fair_shares([len(p) for p in parts], budget)
        parts = [p if len(p) <= c else p[:c] + "\n[... cut to fit ...]" for p, c in zip(parts, caps)]
        head = [PREAMBLE.format(date=datetime.date.today().isoformat())] if preamble else []
        return "\n\n".join(head + parts)

    def step(self, ask: Callable[[str], str], goal: str, direction: str = "", step: str = "", data: str = "") -> str:
        """Choose tools, ask for calls, run them, log everything; return the block for the step's prompt ('' if none ran).

        data: an outline of the attached files (DataBundle.outline()), so the model knows the data exists although the
        tool prompts do not carry it."""
        rec: Dict[str, Any] = dict(step=step, direction=direction)
        if len(self.enabled) > self.max_skills:
            rec["choose_reply"] = ask(self.choose_prompt(goal, direction, data)) or ""
            chosen = parse_choice(rec["choose_reply"], self.enabled, self.max_skills)
            if self.require and not chosen:
                rec["declined_choice"] = True
                again = self.choose_prompt(goal, direction, data) + REQUIRE_CHOICE.format(n=self.max_skills)
                rec["choose_reply_required"] = ask(again) or ""
                chosen = parse_choice(rec["choose_reply_required"], self.enabled, self.max_skills)
                if not chosen:  # declined twice: no tools for this step (never tools the model did not pick)
                    rec["declined_after_require"] = True
        else:
            chosen = list(self.enabled)
        rec["chosen"] = chosen
        results, refused, rec["rounds"] = [], [], []
        for rnd in range(1, self.rounds + 1):
            if not chosen or (rnd > 1 and not results):  # a follow-up needs output to follow up on
                break
            prompt = (
                self.calls_prompt(goal, chosen, direction, data)
                if rnd == 1
                else self.followup_prompt(goal, chosen, results, direction, data)
            )
            reply = ask(prompt) or ""
            ran, no = self._run_reply(reply, chosen, results, rnd)
            entry = dict(round=rnd, reply=reply, results=ran, refused=no)
            # declined, or every call was refused: with require, ask once more for at least one call
            if self.require and rnd == 1 and not ran:
                rec["declined_calls"] = True
                entry["reply_required"] = reply = ask(prompt + REQUIRE_CALL) or ""
                ran2, no2 = self._run_reply(reply, chosen, results, rnd)
                ran, no = ran2, no + no2
                entry.update(results=ran, refused=no)
            rec["rounds"].append(entry)
            results += ran
            refused += no
            if rnd == 1:
                rec["calls_reply"] = entry["reply"]
        rec.update(results=results, refused=refused)
        with self._lock:
            self.log.append(rec)
        return ("\n\n" + self.block(results) + "\n\n") if results else ""

    def _run_reply(self, reply: str, chosen: Sequence[str], earlier: List[Dict[str, Any]], rnd: int):
        """Check and run the calls in one reply; skip a call that repeats one already run. Returns (ran, refused)."""
        seen = {(r["tool"], r["script"], tuple(r["args"])) for r in earlier}
        ran, no = [], []
        for call in _json_array(reply)[: self.max_calls]:
            why = check_call(call, self.catalog, chosen)
            key = None if why else (call["tool"], call["script"], tuple(local_path(str(a)) for a in call["args"]))
            if key in seen:  # in this round or an earlier one
                why = "repeats an earlier call"
            if why:
                no.append(dict(call=call, reason=why))
                continue
            seen.add(key)
            ran.append(dict(self.run_call(call), round=rnd))
        return ran, no
