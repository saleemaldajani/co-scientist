"""Offline tests of the optional web-search step (app/web_search.py) in both modes; no network."""

import json

import app.swarm as swarm
from app.agents import GenerationAgent
from app.models import ContextMemory, ResearchGoal
from app.web_search import WebSearch, _excerpt, _page_text, parse_queries


def fake_search(calls):
    def search(q, k):
        calls.append(q)
        return [
            dict(title=f"Hit {i} for {q}", url=f"https://example.org/{len(calls)}/{i}", snippet="snippet")
            for i in range(k)
        ]

    return search


PAGE = "<html><script>var x=1;</script><p>Lambeth moved its intake to Thames Ditton in 1852, upstream of the sewage.</p><p>Unrelated text about something else entirely here.</p></html>"


def test_parse_queries_json_lines_and_cap():
    assert parse_queries('Sure: ["a b", "A B", "c"]', 3) == ["a b", "c"]
    assert parse_queries("1. first query\n- second query\n", 5) == ["first query", "second query"]
    assert parse_queries('["q1","q2","q3","q4"]', 2) == ["q1", "q2"]
    assert parse_queries("Error: model busy", 3) == []


def test_page_text_and_excerpt():
    t = _page_text(PAGE)
    assert "var x" not in t and "Thames Ditton" in t
    assert "Thames Ditton" in _excerpt(t, "Lambeth intake Thames", 500)
    assert _excerpt(t, "zebra", 500) == ""


def test_cache_shares_results_between_runs(tmp_path):
    calls = []
    path = str(tmp_path / "cache.json")
    w1 = WebSearch(path, k=2, searcher=fake_search(calls), fetcher=lambda u: PAGE)
    r1 = w1.search("Lambeth intake")
    w2 = WebSearch(path, k=2, searcher=fake_search(calls), fetcher=lambda u: (_ for _ in ()).throw(AssertionError()))
    r2 = w2.search("lambeth intake ")
    assert calls == ["Lambeth intake"] and r2["cached"] and r2["hits"] == r1["hits"]
    assert "Thames Ditton" in r1["hits"][0]["excerpt"] and "Thames Ditton" in r1["hits"][1]["excerpt"]


def test_failed_search_is_not_cached_and_step_continues(tmp_path):
    def broken(q, k):
        raise RuntimeError("no results")

    w = WebSearch(str(tmp_path / "c.json"), searcher=broken, fetcher=lambda u: "")
    block = w.step(lambda p: '["x"]', "goal")
    assert "(no results)" in block and w.log[0]["results"][0]["error"].startswith("RuntimeError")
    assert not (tmp_path / "c.json").exists()  # nothing cached after a failure


def test_block_respects_budget():
    calls = []
    w = WebSearch(None, k=5, budget=600, searcher=fake_search(calls), fetcher=lambda u: PAGE * 20)
    b = w.step(lambda p: '["a","b","c"]', "goal")
    assert b.strip().startswith("WEB SEARCH RESULTS") and len(b) < 600 + 600


def test_single_mode_prompt_carries_results_only_when_on(monkeypatch):
    import app.agents as agents

    seen, asked = [], []
    monkeypatch.setattr(
        agents, "call_llm_for_generation", lambda p, **k: seen.append(p) or [{"title": "t", "text": "x"}]
    )
    monkeypatch.setattr(agents, "call_llm", lambda p, **k: asked.append(p) or '["cholera water supply"]')
    web = WebSearch(None, k=1, searcher=fake_search([]), fetcher=lambda u: "")
    GenerationAgent().generate_new_hypotheses(ResearchGoal("Cholera in London", web=web), ContextMemory())
    GenerationAgent().generate_new_hypotheses(ResearchGoal("Cholera in London"), ContextMemory())
    assert "WEB SEARCH RESULTS" in seen[0] and "cholera water supply" in seen[0]
    assert "WEB SEARCH RESULTS" not in seen[1] and len(asked) == 1
    assert web.log[0]["queries"] == ["cholera water supply"]


def test_swarm_each_worker_searches(monkeypatch):
    prompts = []

    def call(prompt, temperature=0.7, model=None):
        prompts.append(prompt)
        if prompt.startswith("You are the orchestrator"):
            return json.dumps(
                [{"name": f"D{i}", "objective": f"o{i}", "data_types": "-", "boundaries": "-"} for i in (1, 2)]
            )
        if "you can search the web" in prompt:
            return '["query for ' + ("D1" if "D1" in prompt else "D2") + '"]'
        if "You are worker" in prompt:
            return json.dumps([{"title": "idea", "text": "text " + str(len(prompts))}])
        if prompt.startswith("You are a panel"):
            return json.dumps({"winner": "A", "reason": "r", "against_B": "b"})
        if prompt.startswith("You are the meta-reviewer"):
            return json.dumps({"critique": [], "strengths": [], "guidance": []})
        return "{}"

    monkeypatch.setattr(swarm, "call_llm", call)
    monkeypatch.setattr(
        swarm, "call_llm_for_reflection", lambda text, temperature=0.5, model=None: {"comment": "ok", "references": []}
    )
    monkeypatch.setattr(swarm, "similarity_score", lambda a, b: 0.0)
    web = WebSearch(None, k=1, searcher=fake_search([]), fetcher=lambda u: "")
    res = swarm.SwarmCoScientist("goal", n_workers=2, per_worker=1, rounds=1, web=web).run()
    workers = [p for p in prompts if "You are worker" in p]
    assert all("WEB SEARCH RESULTS" in p for p in workers)
    assert sorted(q for r in res["web_search"] for q in r["queries"]) == ["query for D1", "query for D2"]
    assert res["settings"]["web"].startswith("web search")
    assert sum(t["role"].endswith("search") for t in res.get("trace", [])) == 2


def test_swarm_unchanged_without_search():
    s = swarm.SwarmCoScientist("goal")
    assert s.web is None and s.web_block(0, {"name": "x"}) == ""
