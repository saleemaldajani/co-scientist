"""Copy the science skills into app/science_skills/ and rebuild their catalog.

    git clone --depth 1 https://github.com/google-deepmind/science-skills /tmp/science-skills
    python scripts/update_science_skills.py /tmp/science-skills

Copies every skill whose scripts run with the app's own dependencies (requests-style HTTP calls through polite-http)
and leaves out the ones that need a model download, an API key with heavy packages, or a desktop program. Then it
writes app/science_skills/catalog.json: for each skill its name, description, scripts, each script's --help text and
the example commands from its SKILL.md. The app reads only the catalog and the scripts, so a run never parses SKILL.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "..", "app", "science_skills")

# Left out, with the reason (shown in app/science_skills/README.md).
SKIP = {
    "alphagenome_atlas_website_links": "needs an AlphaGenome API key and the alphagenome, pandas packages",
    "alphagenome_single_variant_analysis": "needs an AlphaGenome API key and the alphagenome, matplotlib, pandas packages",
    "alphagenome_variant_impact_score": "needs an AlphaGenome API key and the alphagenome, anndata, polars, pysam packages",
    "predictingthepast": "downloads a model and needs JAX",
    "pymol": "drives the PyMOL desktop program",
    "credentials": "instructions for storing API keys, no tool",
    "uv": "instructions for installing the uv package manager, no tool",
    "workflow_skill_creator": "instructions for writing new skills, no tool",
}
# a script name in a command: "uv run scripts/x.py ...", "<SKILL DIR>/scripts/x.py ...", "`x.py --flag <ID>`"
EXAMPLE = re.compile(r"(?:^|[\s`/])([\w-]+\.py)\b([^`|>\n]*)")


def examples_from(md: str, scripts: list) -> list:
    """Example calls (script and arguments) from SKILL.md, with continued lines joined, in order of appearance."""
    text = re.sub(r"\\\n\s*", " ", md)  # backslash line continuations
    out = []
    for line in text.splitlines():
        for m in EXAMPLE.finditer(line):
            script, args = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
            ex = f"{script} {args}".strip()
            if script in scripts and args and ex not in out:
                out.append(ex)
    return out


# The only changes to the copied files: a few scripts tell the services who is calling (STRING's caller_identity, the
# Referer of Europe PMC and OpenAlex requests). Run from this app, they name this app instead of the original collection.
APP_URL = "https://github.com/saleemaldajani/co-scientist"
IDENTITY = [
    (re.compile(r"CALLER_IDENTITY = '[^']*'"), "CALLER_IDENTITY = 'co-scientist-swarm'"),
    (re.compile(r'referer="https://github\.com/[\w.-]+/science-skills/tree/main/[\w-]+"'), f'referer="{APP_URL}"'),
]
NOTICE = "# Modified for co-scientist (scripts/update_science_skills.py): the caller identity names this app.\n"


def name_this_app(path: str) -> bool:
    """Apply IDENTITY to one copied script; mark it as modified (Apache-2.0, section 4b). True if it changed."""
    text = open(path, encoding="utf-8").read()
    new = text
    for pat, rep in IDENTITY:
        new = pat.sub(rep, new)
    if new == text:
        return False
    lines = new.splitlines(keepends=True)
    end = next((i for i, ln in enumerate(lines) if not ln.startswith("#")), 0)  # after the license header
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines[:end]) + NOTICE + "".join(lines[end:]))
    return True


def frontmatter(md: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---", md, re.S)
    if not m:
        return {}
    out, key, buf = {}, None, []
    for line in m.group(1).splitlines():
        top = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if top:
            if key:
                out[key] = " ".join(buf).strip()
            key, rest = top.group(1), top.group(2).strip()
            buf = [] if rest in (">", ">-", "|", "|-", "") else [rest]
        elif key:
            buf.append(line.strip())
    if key:
        out[key] = " ".join(buf).strip()
    return out


def _help(path: str, *sub: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, path, *sub, "-h"], capture_output=True, text=True, timeout=30, cwd=os.path.dirname(path)
        )
        return (r.stdout or r.stderr).strip()
    except Exception as e:  # a script without --help still runs; the examples then carry its usage
        return f"(no --help: {type(e).__name__})"


def script_help(path: str) -> str:
    """The script's --help, then each subcommand's (argparse lists subcommands as {a,b,c} in the usage line)."""
    top = _help(path)
    m = re.search(r"\{([\w,-]+)\}", top.split("\n\n")[0])
    parts = [top[:3000]]
    for cmd in (m.group(1).split(",") if m else [])[:15]:
        h = _help(path, cmd)
        if h.lower().startswith("usage"):
            parts.append(re.sub(r"\n\s*\n", "\n", h)[:1200])
    return "\n\n".join(parts)


def main(src: str) -> None:
    skills_src = os.path.join(src, "skills")
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if os.path.isdir(os.path.join(DEST, "skills")):
        shutil.rmtree(os.path.join(DEST, "skills"))
    os.makedirs(os.path.join(DEST, "skills"), exist_ok=True)
    for f in ("LICENSE", "SKILL_LICENSES.md"):
        shutil.copy(os.path.join(src, f), os.path.join(DEST, f))
    catalog = []
    for d in sorted(os.listdir(skills_src)):
        if d in SKIP or not os.path.isfile(os.path.join(skills_src, d, "SKILL.md")):
            continue
        shutil.copytree(os.path.join(skills_src, d), os.path.join(DEST, "skills", d))
        for root, _, files in os.walk(os.path.join(DEST, "skills", d)):
            for f in files:
                if f.endswith(".py") and name_this_app(os.path.join(root, f)):
                    print(f"  {d}/{f}: caller identity names this app")
        md = open(os.path.join(skills_src, d, "SKILL.md"), encoding="utf-8").read()
        fm = frontmatter(md)
        sdir = os.path.join(DEST, "skills", d, "scripts")
        scripts = sorted(f for f in os.listdir(sdir) if f.endswith(".py")) if os.path.isdir(sdir) else []
        # helper modules (imported by the others, no command line of their own) are not offered to the model
        runnable = [s for s in scripts if "__main__" in open(os.path.join(sdir, s), encoding="utf-8").read()]
        examples = examples_from(md, runnable)
        catalog.append(
            dict(
                name=d,
                title=fm.get("name", d).replace("-", " "),
                description=fm.get("description", ""),
                scripts={s: script_help(os.path.join(sdir, s)) for s in runnable},
                examples=examples[:12],
            )
        )
        print(f"{d}: {len(runnable)} scripts, {len(examples)} examples")
    with open(os.path.join(DEST, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(
            dict(source="https://github.com/google-deepmind/science-skills", commit=commit, skills=catalog), f, indent=1
        )
    with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Science tools\n\n"
            f"Copied from the open-source science-skills collection ({len(catalog)} skills, commit `{commit[:7]}`) by "
            "`scripts/update_science_skills.py`. The skills keep their original files and copyright notices, except that "
            "the scripts that tell a service who is calling (STRING, Europe PMC, OpenAlex) name this app; each of those "
            "files says so below its license header. Scripts and "
            "code are under the Apache License 2.0 (`LICENSE`); the other material is under CC BY 4.0. Each data source "
            "has its own terms of use (`SKILL_LICENSES.md`).\n\n"
            "`catalog.json` is what the app reads: each skill's description, scripts, `--help` text and example commands.\n\n"
            "Not copied:\n\n" + "".join(f"- `{k}`: {v}\n" for k, v in sorted(SKIP.items()))
        )
    print(len(catalog), "skills ->", os.path.normpath(DEST))


if __name__ == "__main__":
    main(sys.argv[1])
