"""Offline tests of the optional simulation check (app/simulation.py) and its two MCP servers; no network."""

import json

import app.sim_servers.records_lab as lab
import app.sim_servers.team_lab as teams
import app.swarm as swarm
from app.simulation import SimulationLab, _json_obj, format_simulation_html, server_choices, server_from_choice

SUB = (
    "window_end,weeks,district,subdistrict,southwark_vauxhall_houses_assigned,southwark_vauxhall_deaths,"
    "lambeth_houses_assigned,lambeth_deaths\n"
    "1854-08-26,7,A,a1,1000,40,1000,5\n1854-08-26,7,B,b1,500,22,1500,6\n"
    "1854-08-26,7,C,c1,2000,70,400,1\n1854-08-26,7,D,d1,800,30,900,4\n"
)
TOTALS = "supply_group,houses_supplied,cholera_deaths\nSouthwark,40046,1263\nLambeth,26107,98\n"


def tables(tmp_path):
    (tmp_path / "south_london_subdistricts_1854.csv").write_text(SUB)
    (tmp_path / "totals.csv").write_text(TOTALS)
    lab.DATA_DIR = str(tmp_path)
    return tmp_path


def test_records_lab_computes_rates_ratios_and_a_held_out_test(tmp_path):
    tables(tmp_path)
    g = {
        x["group"]: x
        for x in lab.group_rates("totals.csv", "supply_group", "cholera_deaths", "houses_supplied")["groups"]
    }
    assert (
        g["Southwark"]["rate"] == 315.387
        and g["Lambeth"]["rate"] == 37.538
        and g["Southwark"]["ratio_to_lowest"] == 8.402
    )
    rr = lab.rate_ratio(1263, 40046, 98, 26107)
    assert rr["ratio"] == 8.402 and rr["ratio_95ci"][0] < 8.402 < rr["ratio_95ci"][1]
    t = lab.held_out_supplier_test()
    assert t["subdistricts_tested"] == 4 and t["supplier_better_in"] == 4
    assert t["supplier_log_loss_per_death"] < t["baseline_log_loss_per_death"]
    assert "error" in lab.elevation_test() and "error" in lab.simulate_handle_removal()  # files absent: say so
    (tmp_path / "soho_daily_outbreak.csv").write_text(
        "date,fatal_cholera_attacks_by_onset,record_type\n"
        + "".join(
            f"1854-09-{d:02d},{v},dated\n" for d, v in zip(range(1, 13), [5, 10, 20, 40, 80, 60, 40, 30, 20, 15, 10, 5])
        )
    )
    rising = lab.simulate_handle_removal(removal_date="1854-09-06")["no_effect_simulation"]
    falling = lab.simulate_handle_removal(removal_date="1854-09-11")["no_effect_simulation"]
    assert rising["projected_onsets_after_removal_if_no_effect"] is None and "rising" in rising["note"]
    assert 0 < falling["projected_onsets_after_removal_if_no_effect"] < 20 and falling["daily_decline_pct"] > 0
    assert "unknown column" in lab.group_rates("totals.csv", "nope", "cholera_deaths", "houses_supplied")["error"]


def test_team_lab_handoffs_decide_which_team_finishes_a_task_first():
    quick = teams.simulate_teams(n_tasks=200, handoff_hours=1, replicates=5)
    slow = teams.simulate_teams(n_tasks=200, handoff_hours=12, replicates=5)
    assert quick["one_task"]["specialists_faster_pct"] > 0 > slow["one_task"]["specialists_faster_pct"]
    assert quick["handoff_hours_at_which_one_task_takes_equally_long"] == 6.0


def test_check_designs_runs_and_evaluates_over_mcp(tmp_path, monkeypatch):
    tables(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    asked = []

    def ask(p):
        asked.append(p)
        if "Write one simulation question" in p:
            call = {
                "tool": "group_rates",
                "arguments": {
                    "file": "totals.csv",
                    "group_column": "supply_group",
                    "events_column": "cholera_deaths",
                    "denominator_column": "houses_supplied",
                },
            }
            return "```json\n" + json.dumps({"question": "Do the rates differ?", "calls": [call]}) + "\n```"
        verdict = "refuted" if "TITLE: first" in p else "supported"
        return json.dumps(
            {
                "verdict": verdict,
                "evidence": "315.387 vs 37.538 per 10,000 houses",
                "revised_title": "",
                "revised_text": "rates 315.387 and 37.538 (group_rates)",
            }
        )

    sim = SimulationLab("records", data_dir=str(tmp_path), top_k=2)
    ideas = [
        dict(id="h1", title="first", text="t1"),
        dict(id="h2", title="second", text="t2"),
        dict(id="h3", title="third", text="t3"),
    ]
    res = sim.check(ask, "goal", ideas, data="- totals.csv: 2 rows")
    assert [e["id"] for e in res["final"]] == ["h2", "h1", "h3"]  # supported first, refuted next, unchecked last
    assert res["final"][0]["verdict"] == "supported" and "315.387" in res["final"][0]["text"]
    assert res["final"][1]["original_text"] == "t1" and res["final"][2] == ideas[2]
    assert all(r["calls_ok"] == 1 and "315.387" in r["results"][0]["output"] for r in res["log"])
    assert "Tool results (computed from the data" in asked[1] and "totals.csv: 2 rows" in asked[0]
    assert "supported" in format_simulation_html(res) and {t["name"] for t in res["tools"]} >= {
        "group_rates",
        "held_out_supplier_test",
    }
    assert all(not k.endswith("_API_KEY") for k in sim._params().env)


def test_check_reports_bad_calls_and_unreadable_replies(tmp_path):
    tables(tmp_path)
    sim = SimulationLab("records", data_dir=str(tmp_path), top_k=1)
    plan = json.dumps({"question": "q", "calls": [{"tool": "no_such_tool", "arguments": {}}]})
    res = sim.check(
        lambda p: plan if "Write one simulation" in p else "no json here", "goal", [dict(id="h1", title="t", text="x")]
    )
    assert res["final"][0]["verdict"] == "inconclusive" and res["log"][0]["calls_ok"] == 0
    assert "unknown tool" in res["log"][0]["results"][0]["output"]


def test_no_verdict_without_a_computed_result_and_one_retry_for_an_unreadable_plan(tmp_path):
    tables(tmp_path)
    asked = []
    good = json.dumps({"question": "q", "calls": [{"tool": "held_out_supplier_test", "arguments": {}}]})

    def ask(p):
        asked.append(p)
        if "Write one simulation" not in p:
            return json.dumps({"verdict": "refuted", "evidence": "made up", "revised_text": "changed"})
        if "TITLE: fixed" in p:
            return '{"calls": [{"tool": "held_out_supplier_test", "arguments: {}}]}' if len(asked) == 1 else good
        return "no plan"

    sim = SimulationLab("records", data_dir=str(tmp_path), top_k=2)
    res = sim.check(ask, "goal", [dict(id="h1", title="fixed", text="t1"), dict(id="h2", title="never", text="t2")])
    by = {e["id"]: e for e in res["final"]}
    assert by["h1"]["verdict"] == "refuted" and res["log"][0]["calls_ok"] == 1  # the retry's plan ran
    assert by["h2"]["verdict"] == "inconclusive" and by["h2"]["title"] == "never"  # nothing ran: not judged
    assert sum("could not be read" in p for p in asked) == 2 and len(asked) == 5  # 2 designs + 2 retries + 1 evaluation


def test_choices_parsing_and_requirements():
    assert server_choices()[0] == "No simulation" and server_from_choice("No simulation") is None
    assert server_from_choice(server_choices()[1]) == "records"
    assert _json_obj('text {"a": {"b": "}"}} more') == {"a": {"b": "}"}}
    try:
        SimulationLab("records")
        raise AssertionError("records lab without data must fail")
    except ValueError:
        pass


def test_swarm_checks_its_final_ranking_only_when_asked(monkeypatch):
    class Stub:
        def describe(self):
            return "stub"

        def check(self, ask, goal, ideas, data="", progress=None):
            return dict(
                server="stub",
                tools=[],
                log=[dict(question="q", results=[], verdict="supported", evidence="e")],
                final=list(reversed(ideas)),
            )

    def llm(prompt, temperature=0.7, model=None):
        if prompt.startswith("You are the orchestrator"):
            return json.dumps(
                [{"name": f"D{i}", "objective": "o", "data_types": "-", "boundaries": "-"} for i in (1, 2)]
            )
        if "You are worker" in prompt:
            return json.dumps([{"title": "idea", "text": "text " + prompt[-20:]}])
        if prompt.startswith("You are a panel"):
            return json.dumps({"winner": "A", "reason": "r", "against_B": "b"})
        if prompt.startswith("You are the meta-reviewer"):
            return json.dumps({"critique": [], "strengths": [], "guidance": []})
        return "{}"

    monkeypatch.setattr(swarm, "call_llm", llm)
    monkeypatch.setattr(
        swarm, "call_llm_for_reflection", lambda text, temperature=0.5, model=None: {"comment": "ok", "references": []}
    )
    monkeypatch.setattr(swarm, "similarity_score", lambda a, b: 0.0)
    res = swarm.SwarmCoScientist("goal", n_workers=2, per_worker=1, rounds=1, sim=Stub()).run()
    assert res["simulation"]["server"] == "stub" and res["settings"]["simulation"] == "stub"
    assert "Simulation check" in swarm.format_swarm_results(res)
    plain = swarm.SwarmCoScientist("goal", n_workers=2, per_worker=1, rounds=1).run()
    assert plain["simulation"] is None and plain["settings"]["simulation"] is None


def test_app_requires_tables_for_the_records_lab():
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("app_main", os.path.join(os.path.dirname(__file__), "..", "app.py"))
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    assert main.build_simulation("No simulation") is None
    try:
        main.build_simulation(server_choices()[1], [])
        raise AssertionError("records lab without tables must fail")
    except ValueError as e:
        assert "attach" in str(e)
    assert main.build_simulation(server_choices()[2]).server == "teams"
