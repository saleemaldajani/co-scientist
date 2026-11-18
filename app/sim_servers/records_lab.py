"""Records lab: an MCP server whose tools compute and simulate from the tables attached to a run.

The co-scientist calls these tools to check its own claims: a rate it states, a comparison it proposes, or a prediction
of what an explanation implies. Every tool reads the CSV files in one folder (the run's attached data) and returns
numbers with the file and columns they came from, so a result can be traced and reproduced.

General tools work on any attached table (describe_tables, group_rates, rate_ratio). The remaining tools are written
for the London cholera records (the files of inputs/london_cholera): a held-out test of the water-supplier
explanation, an elevation test, the change between the 1849 and 1854 epidemics, and a simulation of the Broad Street
handle removal against a no-effect counterfactual. They return an error message when those files are absent.

    python -m app.sim_servers.records_lab <data folder>        # stdio MCP server (the app starts it)
"""

from __future__ import annotations

import csv
import math
import os
import sys
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x (the version gradio[mcp] installs)
    from mcp.server.fastmcp import FastMCP as MCPServer

DATA_DIR = os.environ.get("RECORDS_LAB_DATA") or (sys.argv[1] if len(sys.argv) > 1 else ".")
server = MCPServer("records-lab")


# ---------------------------------------------------------------- reading tables
def _files() -> List[str]:
    return sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv", ".tsv")))


def _read(name: str) -> List[Dict[str, str]]:
    path = os.path.join(DATA_DIR, os.path.basename(name))
    if not os.path.exists(path):
        raise ValueError(f"no table named {name!r}; available: {', '.join(_files())}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t" if path.endswith(".tsv") else ","))


def _num(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").replace(" ", "").strip())
    except (TypeError, ValueError):
        return None


def _need(*names: str) -> Optional[str]:
    missing = [n for n in names if n not in _files()]
    return f"this tool needs {', '.join(missing)}, which is not among the attached tables" if missing else None


# ---------------------------------------------------------------- general tools
@server.tool()
def describe_tables() -> Dict[str, Any]:
    """List the attached tables with their number of rows and their column names."""
    out = []
    for f in _files():
        rows = _read(f)
        out.append(dict(file=f, rows=len(rows), columns=list(rows[0].keys()) if rows else []))
    return dict(data_folder=os.path.basename(os.path.abspath(DATA_DIR)), tables=out)


@server.tool()
def group_rates(
    file: str,
    group_column: str,
    events_column: str,
    denominator_column: str,
    per: float = 10000,
    where_column: str = "",
    where_value: str = "",
) -> Dict[str, Any]:
    """Sum events and denominators by group in one table and return the rate per `per` and the ratio of each group's
    rate to the lowest. Optionally keep only rows where where_column equals where_value. Example: deaths per 10,000
    houses by water company."""
    rows = _read(file)
    if rows and any(c not in rows[0] for c in (group_column, events_column, denominator_column)):
        return dict(error=f"unknown column; {file} has: {', '.join(rows[0].keys())}")
    if where_column:
        rows = [r for r in rows if r.get(where_column, "").strip() == where_value.strip()]
    acc: Dict[str, List[float]] = {}
    for r in rows:
        e, d = _num(r[events_column]), _num(r[denominator_column])
        if e is None or d is None:
            continue
        a = acc.setdefault(r[group_column], [0.0, 0.0, 0])
        a[0] += e
        a[1] += d
        a[2] += 1
    groups = [
        dict(
            group=g, events=round(e, 3), denominator=round(d, 3), rows=int(n), rate=round(e / d * per, 3) if d else None
        )
        for g, (e, d, n) in acc.items()
    ]
    rated = [g for g in groups if g["rate"]]
    low = min((g["rate"] for g in rated), default=None)
    for g in groups:
        g["ratio_to_lowest"] = round(g["rate"] / low, 3) if g["rate"] and low else None
    return dict(file=file, rate_unit=f"{events_column} per {per:g} {denominator_column}", groups=groups)


@server.tool()
def rate_ratio(
    events_a: float, denominator_a: float, events_b: float, denominator_b: float, per: float = 10000
) -> Dict[str, Any]:
    """Rates of two groups, their ratio and a 95% confidence interval for the ratio (log method)."""
    ra, rb = events_a / denominator_a, events_b / denominator_b
    out = dict(rate_a=round(ra * per, 3), rate_b=round(rb * per, 3), per=per, ratio=round(ra / rb, 3) if rb else None)
    if events_a > 0 and events_b > 0 and rb:
        se = math.sqrt(1 / events_a - 1 / denominator_a + 1 / events_b - 1 / denominator_b)
        out["ratio_95ci"] = [round(ra / rb * math.exp(-1.96 * se), 3), round(ra / rb * math.exp(1.96 * se), 3)]
    return out


# ---------------------------------------------------------------- tools for the London cholera records
SUB = "south_london_subdistricts_1854.csv"
EPI = "south_london_1849_1854.csv"
SOHO = "soho_daily_outbreak.csv"


@server.tool()
def held_out_supplier_test() -> Dict[str, Any]:
    """Test the water-supplier explanation by held-out prediction. In each subdistrict supplied by both companies,
    predict how its known-supplier deaths split between Southwark and Vauxhall and Lambeth: the baseline splits them in
    proportion to houses supplied (no supplier effect); the supplier model weights Southwark and Vauxhall houses by
    the death-rate ratio fitted in all other subdistricts. Returns the log loss per death of each model, overall and by
    subdistrict (lower is better)."""
    err = _need(SUB)
    if err:
        return dict(error=err)
    areas = []
    for r in _read(SUB):
        hs, hl = _num(r["southwark_vauxhall_houses_assigned"]), _num(r["lambeth_houses_assigned"])
        ds, dl = _num(r["southwark_vauxhall_deaths"]), _num(r["lambeth_deaths"])
        if hs and hl and hs > 0 and hl > 0 and (ds or 0) + (dl or 0) > 0:
            areas.append(dict(subdistrict=r["subdistrict"], hs=hs, hl=hl, ds=ds or 0, dl=dl or 0))
    out, tot = [], dict(n=0.0, base=0.0, sup=0.0)
    for a in areas:
        rest = [b for b in areas if b is not a]
        ds_, hs_ = sum(b["ds"] for b in rest), sum(b["hs"] for b in rest)
        dl_, hl_ = sum(b["dl"] for b in rest), sum(b["hl"] for b in rest)
        ratio = (ds_ / hs_) / (dl_ / hl_)
        p0 = a["hs"] / (a["hs"] + a["hl"])
        p1 = ratio * a["hs"] / (ratio * a["hs"] + a["hl"])
        n = a["ds"] + a["dl"]
        ll0 = -(a["ds"] * math.log(p0) + a["dl"] * math.log(1 - p0))
        ll1 = -(a["ds"] * math.log(p1) + a["dl"] * math.log(1 - p1))
        tot["n"] += n
        tot["base"] += ll0
        tot["sup"] += ll1
        out.append(
            dict(
                subdistrict=a["subdistrict"],
                deaths=int(n),
                southwark_vauxhall_deaths=int(a["ds"]),
                lambeth_deaths=int(a["dl"]),
                fitted_rate_ratio=round(ratio, 2),
                baseline_log_loss=round(ll0 / n, 3),
                supplier_log_loss=round(ll1 / n, 3),
                supplier_better=ll1 < ll0,
            )
        )
    return dict(
        file=SUB,
        subdistricts_tested=len(out),
        deaths=int(tot["n"]),
        supplier_better_in=sum(o["supplier_better"] for o in out),
        baseline_log_loss_per_death=round(tot["base"] / tot["n"], 3),
        supplier_log_loss_per_death=round(tot["sup"] / tot["n"], 3),
        by_subdistrict=out,
    )


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


@server.tool()
def elevation_test() -> Dict[str, Any]:
    """Test the elevation (miasma) explanation: Spearman correlation between a subdistrict's elevation and its cholera
    deaths per 10,000 residents (1851 census) in the 1849 and 1854 epidemics, across all subdistricts and within each
    water-company service group."""
    err = _need(EPI)
    if err:
        return dict(error=err)
    rows = [r for r in _read(EPI) if _num(r["population_1851_census"])]

    def corr(rs, col):
        el = [_num(r["elevation_ft_above_trinity_high_water_mark"]) for r in rs]
        mort = [_num(r[col]) / _num(r["population_1851_census"]) * 10000 for r in rs]
        keep = [(a, b) for a, b in zip(el, mort) if a is not None and b is not None]
        return dict(subdistricts=len(keep), spearman_rho=_spearman([a for a, _ in keep], [b for _, b in keep]))

    out = dict(
        file=EPI,
        all_subdistricts={
            "1849": corr(rows, "cholera_deaths_1849"),
            "1854": corr(rows, "cholera_deaths_1854_through_october_21"),
        },
    )
    groups: Dict[str, Any] = {}
    for g in sorted({r["water_company_service_group_1854"] for r in rows}):
        rs = [r for r in rows if r["water_company_service_group_1854"] == g]
        groups[g] = {
            "1849": corr(rs, "cholera_deaths_1849"),
            "1854": corr(rs, "cholera_deaths_1854_through_october_21"),
        }
    out["by_service_group"] = groups
    return out


@server.tool()
def epidemic_change() -> Dict[str, Any]:
    """Deaths in the 1849 and 1854 epidemics by the subdistricts' 1854 water-company service group, with the change
    between them. (Lambeth moved its intake upstream in 1852; Southwark and Vauxhall did not.)"""
    err = _need(EPI)
    if err:
        return dict(error=err)
    acc: Dict[str, List[float]] = {}
    for r in _read(EPI):
        a = acc.setdefault(r["water_company_service_group_1854"], [0, 0, 0, 0])
        a[0] += _num(r["cholera_deaths_1849"]) or 0
        a[1] += _num(r["cholera_deaths_1854_through_october_21"]) or 0
        a[2] += _num(r["population_1851_census"]) or 0
        a[3] += 1
    return dict(
        file=EPI,
        note="1854 deaths are counted through 21 October; 1849 covers the whole epidemic",
        groups=[
            dict(
                service_group=g,
                subdistricts=int(n),
                deaths_1849=int(d49),
                deaths_1854=int(d54),
                change=int(d54 - d49),
                per_10000_1849=round(d49 / p * 1e4, 1) if p else None,
                per_10000_1854=round(d54 / p * 1e4, 1) if p else None,
            )
            for g, (d49, d54, p, n) in acc.items()
        ],
    )


@server.tool()
def simulate_handle_removal(removal_date: str = "1854-09-08", fit_days: int = 5) -> Dict[str, Any]:
    """Simulate the Broad Street outbreak without the handle removal. Fits an exponential decline to the dated fatal
    onsets in the fit_days before removal_date, projects it forward as the no-effect counterfactual, and compares the
    projected onsets after removal with the recorded ones. Also reports the peak and whether onsets were already
    falling before the removal. Onsets with unknown dates are excluded (their count is reported)."""
    err = _need(SOHO)
    if err:
        return dict(error=err)
    rows = _read(SOHO)
    dated = [
        (date.fromisoformat(r["date"]), _num(r["fatal_cholera_attacks_by_onset"]) or 0)
        for r in rows
        if r.get("record_type", "dated") == "dated" and r.get("date")
    ]
    unknown = sum(
        _num(r["fatal_cholera_attacks_by_onset"]) or 0 for r in rows if r.get("record_type") not in ("dated", None)
    )
    dated.sort()
    removal = date.fromisoformat(removal_date)
    peak_day, peak = max(dated, key=lambda x: x[1])
    on_day = sum(v for d, v in dated if d == removal)
    before = sum(v for d, v in dated if d < removal)
    after = sum(v for d, v in dated if d > removal)
    fit = [(d, v) for d, v in dated if removal - timedelta(days=fit_days) <= d < removal and v > 0]
    sim = None
    if len(fit) >= 3:
        xs = [(d - removal).days for d, _ in fit]
        ys = [math.log(v) for _, v in fit]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        icpt = my - slope * mx
        last = max(d for d, _ in dated)
        proj = sum(math.exp(icpt + slope * k) for k in range(1, (last - removal).days + 1))
        sim = dict(
            fit_window=f"{fit[0][0]} to {fit[-1][0]}",
            daily_decline_pct=round((1 - math.exp(slope)) * 100, 1),
            projected_onsets_after_removal_if_no_effect=round(proj, 1),
            recorded_onsets_after_removal=int(after),
        )
    return dict(
        file=SOHO,
        peak_date=str(peak_day),
        peak_onsets=int(peak),
        removal_date=removal_date,
        onsets_on_removal_day=int(on_day),
        dated_onsets_before_removal=int(before),
        dated_onsets_after_removal=int(after),
        unknown_onset_dates_excluded=int(unknown),
        declining_before_removal=on_day < peak and peak_day < removal,
        no_effect_simulation=sim,
    )


if __name__ == "__main__":
    server.run()
