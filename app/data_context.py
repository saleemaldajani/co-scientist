"""Uploaded data for a run: a budgeted extract for each prompt, and a check of the numbers a hypothesis quotes.

Models see only text, and a local model's context holds a few tens of thousands of tokens, so the files are not pasted
whole. Each step gets an extract that fits a character budget:
  - text files (README, data dictionaries) first, truncated if needed;
  - small tables whole;
  - large tables as their header plus the rows most relevant to that step (word overlap with the step's query,
    weighted by how rare each word is in the table), so a worker on sleep sees sleep rows and a judge sees the rows
    behind the two hypotheses it compares.
After a run, every number a hypothesis quotes is looked up in the files (check_numbers), so the reader can see which
numbers came from the data and which did not.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

TABLE_EXT = {".csv": ",", ".tsv": "\t"}
TEXT_EXT = {".md", ".txt", ".json", ".yaml", ".yml"}
MAX_FILE_BYTES = 60 * 1024 * 1024
STOP = set(
    "the and for with that this from into are was were has have had not but all any can each per its their there these those "
    "which what when where who how than then them they will would should could may might also only more most less such use used "
    "using data table tables file files column columns row rows number numbers hypothesis hypotheses research goal study".split()
)
WORD = re.compile(r"[a-z][a-z0-9_]{2,}")
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
PREAMBLE = (
    "ATTACHED DATA - an extract of {n} file(s) the user uploaded, read-only. There is no sandbox, file system or tool: "
    "this extract is all the data you can see, so do not write code or commands and do not ask to open files. "
    "Quote numbers only from this extract and name the file (and row) each comes from; write 'not in the extract' "
    "for any number you cannot find here. Large tables show only the rows most relevant to this step."
)


def _words(t: str) -> List[str]:
    return [w for w in WORD.findall(t.lower()) if w not in STOP]


def _norm_numbers(t: str) -> str:
    """'180 000' / '180 000' (thin, narrow or no-break space thousands) -> '180,000'."""
    return re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", ",", t)


@dataclass
class DataFile:
    name: str
    kind: str  # "table" or "text"
    text: str = ""
    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def row_line(self, r: Sequence[str]) -> str:
        return ",".join((c if len(c) <= 80 else c[:77] + "...").replace("\n", " ") for c in r)

    def full_csv(self) -> str:
        return "\n".join([",".join(self.header)] + [self.row_line(r) for r in self.rows])


class DataBundle:
    """The files uploaded for one run."""

    def __init__(self, files: List[DataFile], skipped: Optional[List[str]] = None):
        self.files = files
        self.skipped = skipped or []
        self._idf: Dict[str, Tuple[Dict[str, float], List[Counter]]] = {}
        self._numbers: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_paths(cls, paths: Sequence[str]) -> "DataBundle":
        files, skipped = [], []
        for p in paths or []:
            p = str(p)
            name = os.path.basename(p)
            ext = os.path.splitext(name)[1].lower()
            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    skipped.append(f"{name}: larger than {MAX_FILE_BYTES // 2**20} MB")
                    continue
                raw = open(p, encoding="utf-8", errors="replace").read()
            except OSError as e:
                skipped.append(f"{name}: {e}")
                continue
            if ext in TABLE_EXT:
                rd = list(csv.reader(io.StringIO(raw), delimiter=TABLE_EXT[ext]))
                if not rd:
                    skipped.append(f"{name}: empty")
                    continue
                files.append(
                    DataFile(name, "table", header=rd[0], rows=[r for r in rd[1:] if any(c.strip() for c in r)])
                )
            elif ext in TEXT_EXT:
                files.append(DataFile(name, "text", text=raw))
            else:
                skipped.append(f"{name}: unsupported type (use .csv, .tsv, .md, .txt or .json)")
        # text files first (READMEs explain the tables), then tables by name
        files.sort(key=lambda f: (f.kind != "text", not f.name.lower().startswith("readme"), f.name.lower()))
        return cls(files, skipped)

    @classmethod
    def from_dir(cls, folder: str) -> "DataBundle":
        return cls.from_paths(sorted(os.path.join(folder, f) for f in os.listdir(folder) if not f.startswith(".")))

    def __bool__(self) -> bool:
        return bool(self.files)

    def summary(self) -> str:
        parts = [
            f"{f.name} ({f.n_rows:,} rows × {len(f.header)} columns)"
            if f.kind == "table"
            else f"{f.name} ({len(f.text):,} characters)"
            for f in self.files
        ]
        out = f"{len(self.files)} file(s): " + "; ".join(parts)
        if self.skipped:
            out += " | skipped: " + "; ".join(self.skipped)
        return out

    def outline(self, max_chars: int = 2000) -> str:
        """One line per file (name, size, column names): what a step that does not see the data needs to know of it."""
        lines = []
        for f in self.files:
            if f.kind == "table":
                cols = ", ".join(f.header)
                if len(cols) > 300:
                    cols = cols[:297] + "..."
                lines.append(f"- {f.name}: {f.n_rows:,} rows; columns: {cols}")
            else:
                lines.append(f"- {f.name}: text, {len(f.text):,} characters")
        out = "\n".join(lines)
        return out if len(out) <= max_chars else out[: max_chars - 25] + "\n[... more files ...]"

    # ------------------------------------------------------------------ extract for a prompt
    def _relevance(self, f: DataFile, query: str) -> List[float]:
        if f.name not in self._idf:
            docs = [Counter(_words(" ".join(r))) for r in f.rows]
            df = Counter(w for d in docs for w in d)
            n = max(1, len(docs))
            self._idf[f.name] = ({w: math.log(1 + n / c) for w, c in df.items()}, docs)
        idf, docs = self._idf[f.name]
        q = set(_words(query))
        return [sum(idf.get(w, 0.0) for w in q if w in d) for d in docs]

    def _count_col(self, f: DataFile) -> Optional[int]:
        for i, h in enumerate(f.header):
            if h.lower() in ("n_participants", "participants", "count", "n", "n_persons", "person_count"):
                return i
        return None

    def _table_extract(self, f: DataFile, query: str, budget: int) -> str:
        head = f"### {f.name} ({f.n_rows:,} rows × {len(f.header)} columns)\n"
        full = f.full_csv()
        if len(head) + len(full) <= budget:
            return head + full
        rel = self._relevance(f, query)
        ci = self._count_col(f)

        def size(r):
            try:
                return float(r[ci].replace(",", "")) if ci is not None and ci < len(r) else 0.0
            except ValueError:
                return 0.0

        order = sorted(range(f.n_rows), key=lambda i: (-rel[i], -size(f.rows[i]), i))
        lines, used = [",".join(f.header)], len(head) + len(",".join(f.header)) + 120
        for i in order:
            ln = f.row_line(f.rows[i])
            if used + len(ln) + 1 > budget:
                break
            lines.append(ln)
            used += len(ln) + 1
        shown = len(lines) - 1
        note = f"(showing {shown:,} of {f.n_rows:,} rows: those most relevant to this step, then the largest; the other rows exist but are not shown)"
        return head + note + "\n" + "\n".join(lines)

    def brief(self, query: str = "", budget: int = 20000) -> str:
        """An extract of every file that fits `budget` characters, chosen for `query`."""
        if not self.files:
            return ""
        pre = PREAMBLE.format(n=len(self.files))
        left = max(1000, budget - len(pre) - 50)
        texts = [f for f in self.files if f.kind == "text"]
        tables = [f for f in self.files if f.kind == "table"]
        out = [pre]
        # text files: up to 35% of the budget (all of it if there are no tables), split evenly
        if texts:
            share = int(left * (0.35 if tables else 1.0))
            each = share // len(texts)
            for f in texts:
                body = f.text if len(f.text) <= each else f.text[: max(0, each - 60)] + "\n[... truncated ...]"
                out.append(f"### {f.name}\n{body}")
                left -= len(out[-1]) + 2
        # tables: water-filling, so small tables go in whole and large ones share what is left
        if tables:
            need = {f.name: len(f.full_csv()) + 80 for f in tables}
            alloc, rest, pending = {}, max(0, left), sorted(tables, key=lambda f: need[f.name])
            while pending:
                fair = rest // len(pending)
                f = pending[0]
                if need[f.name] <= fair:
                    alloc[f.name] = need[f.name]
                    rest -= need[f.name]
                    pending.pop(0)
                else:
                    for g in pending:
                        alloc[g.name] = fair
                    break
            for f in tables:
                if alloc.get(f.name, 0) >= min(200, need[f.name]):
                    out.append(self._table_extract(f, query, alloc[f.name]))
                else:
                    out.append(
                        f"### {f.name} ({f.n_rows:,} rows; columns: {', '.join(f.header)[:300]}) - no room in this step's extract"
                    )
        return "\n\n".join(out)

    # ------------------------------------------------------------------ number check
    def _number_index(self) -> Dict[str, List[str]]:
        if self._numbers is None:
            idx: Dict[str, List[str]] = {}
            for f in self.files:
                if f.kind == "table":
                    for i, r in enumerate(f.rows, 1):
                        label = " | ".join(c for c in r[:3] if c)[:80]
                        for j, c in enumerate(r):
                            for m in NUM.finditer(_norm_numbers(c)):
                                key = m.group(1).replace(",", "") + (m.group(2) or "")
                                col = f.header[j] if j < len(f.header) else f"col{j + 1}"
                                idx.setdefault(key, []).append(f"{f.name}, row {i} ({label}), {col}")
                else:
                    for k, line in enumerate(f.text.splitlines(), 1):
                        for m in NUM.finditer(_norm_numbers(line)):
                            key = m.group(1).replace(",", "") + (m.group(2) or "")
                            idx.setdefault(key, []).append(f"{f.name}, line {k}")
            self._numbers = idx
        return self._numbers

    def check_numbers(self, text: str, min_value: float = 20) -> List[Dict[str, object]]:
        """Every number of at least `min_value` in `text` (years 1900-2100 left out), with where it appears in the files."""
        idx = self._number_index()
        seen, out = set(), []
        for m in NUM.finditer(_norm_numbers(text or "")):
            key = m.group(1).replace(",", "") + (m.group(2) or "")
            try:
                v = float(key)
            except ValueError:
                continue
            if v < min_value or (m.group(2) is None and 1900 <= v <= 2100) or key in seen:
                continue
            seen.add(key)
            where = idx.get(key, [])
            out.append(dict(number=m.group(0), found=bool(where), where=where[:3], n_places=len(where)))
        return out


def data_check_summary(checks: List[Dict[str, object]]) -> str:
    if not checks:
        return "no numbers quoted"
    found = sum(1 for c in checks if c["found"])
    return f"{found} of {len(checks)} quoted numbers found in the uploaded data"


def data_check_html(checks: List[Dict[str, object]]) -> str:
    """Small HTML block for the results panel; neutral colors so it reads in light and dark mode."""
    import html as H

    if not checks:
        return '<p style="opacity:.8"><b>Data check:</b> no numbers quoted.</p>'
    items = []
    for c in checks:
        mark = "✅" if c["found"] else "❌"
        where = H.escape("; ".join(c["where"])) if c["found"] else "not in the uploaded data"
        more = f" (+{c['n_places'] - len(c['where'])} more places)" if c["n_places"] > len(c["where"]) else ""
        items.append(f"<li>{mark} <code>{H.escape(str(c['number']))}</code>: {where}{more}</li>")
    return (
        f'<details style="margin:.4em 0"><summary><b>Data check:</b> {H.escape(data_check_summary(checks))}</summary>'
        f'<ul style="margin:.3em 0 .3em 1.2em">{"".join(items)}</ul>'
        '<p style="opacity:.75;font-size:.9em">A number found in the data may still come from a different row than the one the '
        "hypothesis means; check the location.</p></details>"
    )
