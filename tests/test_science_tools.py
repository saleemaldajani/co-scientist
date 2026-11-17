"""Offline tests of the optional science-tools step (app/science_tools.py) in both modes; no network."""

import json
import os

import app.swarm as swarm
from app.agents import GenerationAgent
from app.models import ContextMemory, ResearchGoal
from app.science_tools import (
    SKILLS_DIR,
    ScienceTools,
    check_call,
    default_runner,
    load_catalog,
    parse_choice,
    tool_choices,
)

CATALOG = load_catalog()


def fake_runner(ran):
    def run(path, args, timeout):
        ran.append((os.path.basename(path), args))
        return dict(returncode=0, output=f"P04637 TP53 result for {' '.join(args)}")

    return run


def test_catalog_has_the_copied_skills_and_their_scripts():
    assert len(CATALOG) >= 30
    for name, s in CATALOG.items():
        assert s["description"] and s["scripts"], name
        for script in s["scripts"]:
            assert os.path.isfile(os.path.join(SKILLS_DIR, "skills", name, "scripts", script)), (name, script)
    assert "uniprot_database" in CATALOG and "pubmed_database" in CATALOG
    labels = dict((v, k) for k, v in tool_choices())
    assert labels["uniprot_database"] == "UniProt proteins" and len(labels) == len(CATALOG)


def test_parse_choice_keeps_only_enabled_tools_in_order():
    allowed = ["uniprot_database", "pubmed_database", "chembl_database"]
    assert parse_choice('Sure: ["pubmed-database", "UniProt_database", "pdb_database"]', allowed, 2) == [
        "pubmed_database",
        "uniprot_database",
    ]
    assert parse_choice("no json here", allowed, 2) == []


def test_check_call_refuses_unknown_tools_scripts_and_paths():
    en = ["uniprot_database"]
    ok = dict(tool="uniprot_database", script="uniprot_tools.py", args=["get", "P04637"])
    assert check_call(ok, CATALOG, en) is None
    assert "not enabled" in check_call(dict(ok, tool="pubmed_database"), CATALOG, en)
    assert "no script" in check_call(dict(ok, script="../../app.py"), CATALOG, en)
    assert "not allowed" in check_call(dict(ok, args=["--out", "/etc/passwd"]), CATALOG, en)
    assert "not allowed" in check_call(dict(ok, args=["../x"]), CATALOG, en)
    assert "list of strings" in check_call(dict(ok, args="get P04637"), CATALOG, en)


def test_step_chooses_then_runs_calls_and_logs():
    ran, asked = [], []

    def ask(p):
        asked.append(p)
        if "Choose up to" in p:
            return '["uniprot_database"]'
        if "follow-up calls" in p:  # round 2 sees round 1's output and asks for the details it points to
            assert "P04637 TP53 result for get P04637" in p
            return json.dumps([{"tool": "uniprot_database", "script": "uniprot_tools.py", "args": ["get", "P38398"]}])
        return json.dumps(
            [
                {"tool": "uniprot_database", "script": "uniprot_tools.py", "args": ["get", "P04637"]},
                {"tool": "pubmed_database", "script": "pubmed_api.py", "args": ["search", "TP53"]},  # not chosen
            ]
        )

    t = ScienceTools(catalog=CATALOG, runner=fake_runner(ran))  # every tool enabled
    block = t.step(ask, "Find drivers of tumour suppression", step="generation 1")
    assert block.strip().startswith("TOOL RESULTS") and "uniprot_tools.py get P04637" in block
    assert ran == [("uniprot_tools.py", ["get", "P04637"]), ("uniprot_tools.py", ["get", "P38398"])]
    assert "uniprot_tools.py" in asked[1] and "usage" in asked[1].lower()  # the chosen tool's --help is shown
    assert len(asked) == 3 and "round 2 · uniprot_database: uniprot_tools.py get P38398" in block
    rec = t.log[0]
    assert rec["chosen"] == ["uniprot_database"] and len(rec["results"]) == 2 and len(rec["refused"]) == 1
    assert [len(r["results"]) for r in rec["rounds"]] == [1, 1]


def test_followup_skips_repeats_and_runs_only_after_output():
    ran, asked = [], []
    call = {"tool": "uniprot_database", "script": "uniprot_tools.py", "args": ["get", "P04637"]}
    t = ScienceTools(["uniprot_database"], catalog=CATALOG, runner=fake_runner(ran))
    t.step(lambda p: asked.append(p) or json.dumps([call, call]), "goal")  # the same call twice, then again in round 2
    assert len(ran) == 1 and len(asked) == 2
    assert [r["reason"] for r in t.log[0]["refused"]] == ["repeats an earlier call"] * 3  # 1 in round 1, 2 in round 2
    asked.clear()
    t.step(lambda p: asked.append(p) or "[]", "goal")  # nothing ran in round 1: no follow-up call
    assert len(asked) == 1
    one = ScienceTools(["uniprot_database"], catalog=CATALOG, runner=fake_runner([]), rounds=1)
    asked.clear()
    one.step(lambda p: asked.append(p) or json.dumps([call]), "goal")
    assert len(asked) == 1 and len(one.log[0]["rounds"]) == 1


def test_block_shares_the_budget_between_calls():
    t = ScienceTools(["uniprot_database"], catalog=CATALOG, budget=1000)
    res = [
        dict(tool="uniprot_database", script="uniprot_tools.py", args=[str(i)], returncode=0, output="y" * 900, round=i)
        for i in (1, 2)
    ]
    b = t.block(res)
    assert b.count("cut to fit") == 2 and "round 2 · uniprot_database" in b


def test_few_enabled_tools_skip_the_choice_call():
    asked = []
    t = ScienceTools(["uniprot_database"], catalog=CATALOG, runner=fake_runner([]))
    t.step(lambda p: asked.append(p) or "[]", "goal")
    assert len(asked) == 1 and "Choose up to" not in asked[0]
    assert t.log[0]["results"] == []


def test_output_and_block_are_cut_to_budget():
    def big(path, args, timeout):
        return dict(returncode=0, output="x" * 50_000)

    t = ScienceTools(["uniprot_database"], catalog=CATALOG, runner=big, output_chars=1000, budget=1500)
    call = {"tool": "uniprot_database", "script": "uniprot_tools.py", "args": ["get", "P04637"]}
    b = t.step(lambda p: json.dumps([call, call, call]), "goal")
    assert "more characters cut" in b and len(b) < 1500 + 400


def test_runner_runs_a_real_script_without_network():
    script = os.path.join(SKILLS_DIR, "skills", "uniprot_database", "scripts", "uniprot_tools.py")
    r = default_runner(script, ["-h"], 60)
    assert r["returncode"] == 0 and "usage" in r["output"].lower()


def test_runner_does_not_pass_the_model_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    probe = tmp_path / "probe.py"
    probe.write_text("import os; print(os.environ.get('OPENROUTER_API_KEY', 'absent'))")
    assert default_runner(str(probe), [], 30)["output"].strip() == "absent"


def test_single_mode_prompt_carries_tool_results_only_when_on(monkeypatch):
    import app.agents as agents

    seen = []
    monkeypatch.setattr(
        agents, "call_llm_for_generation", lambda p, **k: seen.append(p) or [{"title": "t", "text": "x"}]
    )
    call = {"tool": "uniprot_database", "script": "uniprot_tools.py", "args": ["get", "P04637"]}
    monkeypatch.setattr(agents, "call_llm", lambda p, **k: json.dumps([call]))
    tools = ScienceTools(["uniprot_database"], catalog=CATALOG, runner=fake_runner([]))
    GenerationAgent().generate_new_hypotheses(ResearchGoal("TP53 in cancer", tools=tools), ContextMemory())
    GenerationAgent().generate_new_hypotheses(ResearchGoal("TP53 in cancer"), ContextMemory())
    assert "TOOL RESULTS" in seen[0] and "P04637 TP53" in seen[0]
    assert "TOOL RESULTS" not in seen[1]


def test_swarm_each_worker_uses_tools(monkeypatch):
    prompts = []
    call = {"tool": "uniprot_database", "script": "uniprot_tools.py", "args": ["get", "P04637"]}

    def llm(prompt, temperature=0.7, model=None):
        prompts.append(prompt)
        if prompt.startswith("You are the orchestrator"):
            return json.dumps(
                [{"name": f"D{i}", "objective": f"o{i}", "data_types": "-", "boundaries": "-"} for i in (1, 2)]
            )
        if "you can run these scientific tools" in prompt:
            return json.dumps([call])
        if "You are worker" in prompt:
            return json.dumps([{"title": "idea", "text": "text " + str(len(prompts))}])
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
    tools = ScienceTools(["uniprot_database"], catalog=CATALOG, runner=fake_runner([]))
    res = swarm.SwarmCoScientist("goal", n_workers=2, per_worker=1, rounds=1, tools=tools).run()
    workers = [p for p in prompts if "You are worker" in p]
    assert len(workers) == 2 and all("TOOL RESULTS" in p for p in workers)
    assert len(res["tool_calls"]) == 2 and res["settings"]["tools"].startswith("science tools")
    assert sum(t["role"].endswith("tools") for t in res.get("trace", [])) == 4  # calls + follow-up, per worker


def test_swarm_unchanged_without_tools():
    s = swarm.SwarmCoScientist("goal")
    assert s.tools is None and s.tools_block(0, {"name": "x"}) == ""


def test_build_tools_modes():
    """The demo's three choices: No tools (the default) changes nothing; All tools; or the ticked tools only."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("app_main", pathlib.Path(__file__).parents[1] / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_tools("No tools", ["uniprot_database"]) == (None, None)
    tools, web = mod.build_tools("All tools")
    assert len(tools.enabled) == len(CATALOG) and web is not None
    tools, web = mod.build_tools("Choose tools", ["web_search"])
    assert tools is None and web is not None
    tools, web = mod.build_tools("Choose tools", ["pubmed_database", "uniprot_database"])
    assert tools.enabled == ["pubmed_database", "uniprot_database"] and web is None
    assert mod.build_tools("Choose tools", []) == (None, None)


def test_output_files_come_back_and_tmp_paths_stay_in_the_scratch_folder(tmp_path):
    from app.science_tools import local_path

    assert local_path("/tmp/out.json") == "out.json" and local_path("/tmp/") == "/tmp/"
    assert (
        check_call(
            dict(tool="uniprot_database", script="uniprot_tools.py", args=["-o", "/tmp/x.json"]),
            CATALOG,
            ["uniprot_database"],
        )
        is None
    )
    probe = tmp_path / "writer.py"
    probe.write_text("import sys; open(sys.argv[1], 'w').write('{\"hits\": 3}'); print('Success! Data written')")
    r = default_runner(str(probe), [local_path("/tmp/res.json")], 30)
    assert r["returncode"] == 0 and "[file res.json]" in r["output"] and '{"hits":3}' in r["output"]
    assert not os.path.exists("/tmp/res.json") or os.path.getmtime("/tmp/res.json") < os.path.getmtime(probe)


def test_fair_shares_keep_short_outputs_whole():
    from app.science_tools import fair_shares

    assert fair_shares([100, 200], 1000) == [100, 200]  # everything fits
    caps = fair_shares([250, 150, 4000, 6000, 6000], 12000)
    assert caps[:2] == [250, 150] and sum(caps) <= 12000 and max(caps[2:]) - min(caps[2:]) <= 1
