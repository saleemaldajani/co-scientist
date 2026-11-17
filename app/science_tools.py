"""Optional science tools before each generation step, for single and swarm mode alike.

The tools are the scripts in app/science_skills/ (databases such as UniProt, PubMed, ChEMBL, Ensembl, ClinVar and
ClinicalTrials.gov, and literature search). When a run has tools on, every step that writes new hypotheses (a
single-mode generation call; each swarm worker) first:

1. asks the model which of the enabled tools would help (skipped when `max_skills` or fewer are enabled),
2. shows the model those tools' command lines (each script's --help text and the examples from its SKILL.md) and asks
   for up to `max_calls` calls as JSON: {"tool": ..., "script": ..., "args": [...]},
3. runs each call as `python <script> <args>` (no shell, a time limit, a scratch folder, only scripts of enabled
   tools), and puts the output into that step's prompt as a TOOL RESULTS block, cut to a character budget.

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
                    parts.append(f"[file {rel}]\n{fh.read()}")
            except UnicodeDecodeError:
                parts.append(f"[file {rel}: {size:,} bytes, not shown]")
    return "\n".join(parts)


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


class ScienceTools:
    def __init__(
        self,
        enabled: Optional[Sequence[str]] = None,
        max_skills: int = 2,
        max_calls: int = 3,
        timeout: float = 60.0,
        output_chars: int = 4000,
        budget: int = 8000,
        usage_chars: int = 8000,
        catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        runner: Optional[Callable[[str, List[str], float], Dict[str, Any]]] = None,
        skills_dir: str = SKILLS_DIR,
    ):
        """enabled: tool names (folder names in app/science_skills/skills); None or empty = every tool."""
        self.catalog = load_catalog() if catalog is None else catalog
        unknown = [e for e in (enabled or []) if e not in self.catalog]
        if unknown:
            raise ValueError(f"unknown science tools: {unknown}")
        self.enabled = list(enabled) if enabled else sorted(self.catalog)
        self.max_skills, self.max_calls, self.timeout = max_skills, max_calls, timeout
        self.output_chars, self.budget, self.usage_chars = output_chars, budget, usage_chars
        self.runner = runner or default_runner
        self.skills_dir = skills_dir
        self.log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return bool(self.enabled)

    def describe(self) -> str:
        n = len(self.enabled)
        which = "all" if n == len(self.catalog) else ", ".join(self.enabled)
        return (
            f"science tools before each generation step ({n} enabled: {which}): the model picks up to {self.max_skills} "
            f"tools and writes up to {self.max_calls} calls, each limited to {self.timeout:.0f} s; about {self.budget:,} "
            "characters of output go into the prompt"
        )

    # ------------------------------------------------------------------ prompts
    def _head(self, goal: str, direction: str) -> str:
        return f"Research Goal: {goal}\n\n" + (f"Your direction: {direction}\n\n" if direction else "")

    def choose_prompt(self, goal: str, direction: str = "") -> str:
        lines = [f"- {n}: {self.catalog[n]['description']}" for n in self.enabled]
        return (
            self._head(goal, direction)
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

    def calls_prompt(self, goal: str, chosen: Sequence[str], direction: str = "") -> str:
        return (
            self._head(goal, direction)
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

    # ------------------------------------------------------------------ running
    def run_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        tool, script, args = call["tool"], call["script"], [local_path(str(a)) for a in call.get("args", [])]
        path = os.path.join(self.skills_dir, "skills", tool, "scripts", script)
        res = self.runner(path, args, self.timeout)
        out = (res.get("output") or "").strip()
        if len(out) > self.output_chars:
            out = out[: self.output_chars] + f"\n[... {len(out) - self.output_chars:,} more characters cut ...]"
        return dict(tool=tool, script=script, args=args, returncode=res.get("returncode"), output=out)

    def block(self, results: List[Dict[str, Any]]) -> str:
        out, used = [PREAMBLE.format(date=datetime.date.today().isoformat())], 0
        for r in results:
            cmd = " ".join([r["script"], *[json.dumps(a) if " " in a else a for a in r["args"]]])
            status = "" if r["returncode"] == 0 else f" (exit code {r['returncode']})"
            part = f"### {r['tool']}: {cmd}{status}\n{r['output'] or '(no output)'}"
            if used + len(part) > self.budget:
                part = part[: max(0, self.budget - used)] + "\n[... cut to fit ...]"
            out.append(part)
            used += len(part)
            if used >= self.budget:
                break
        return "\n\n".join(out)

    def step(self, ask: Callable[[str], str], goal: str, direction: str = "", step: str = "") -> str:
        """Choose tools, ask for calls, run them, log everything; return the block for the step's prompt ('' if none ran)."""
        rec: Dict[str, Any] = dict(step=step, direction=direction)
        if len(self.enabled) > self.max_skills:
            rec["choose_reply"] = ask(self.choose_prompt(goal, direction)) or ""
            chosen = parse_choice(rec["choose_reply"], self.enabled, self.max_skills)
        else:
            chosen = list(self.enabled)
        rec["chosen"] = chosen
        results, refused = [], []
        if chosen:
            rec["calls_reply"] = ask(self.calls_prompt(goal, chosen, direction)) or ""
            for call in _json_array(rec["calls_reply"])[: self.max_calls]:
                why = check_call(call, self.catalog, chosen)
                if why:
                    refused.append(dict(call=call, reason=why))
                    continue
                results.append(self.run_call(call))
        rec.update(results=results, refused=refused)
        with self._lock:
            self.log.append(rec)
        return ("\n\n" + self.block(results) + "\n\n") if results else ""
