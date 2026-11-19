"""Offline tests of research-group mode (app/research_group.py): the LLM boundary is mocked, no key and no network."""

import json
import re

import pytest

import app.research_group as rg
import app.swarm as swarm


def fake_llm():
    calls = []

    def call(prompt, temperature=0.7, model=None):
        calls.append((model, prompt))
        if "agenda: choose a focus" in prompt:
            n = int(re.search(r"group with (\d+) student", prompt).group(1))
            p = re.search(r"You are PI (\d+)", prompt).group(1)
            return json.dumps(
                {
                    "focus": f"focus {p}",
                    "briefing": "Southwark and Vauxhall: 1263 deaths.",
                    "projects": [
                        {"name": f"P{p}-{i}", "objective": "o", "data_types": "t", "boundaries": "b"} for i in range(n)
                    ],
                }
            )
        if "You are a student" in prompt or "Your project:" in prompt and "Propose" in prompt:
            proj = re.search(r"Your project: (\S+)\.", prompt).group(1)
            n = int(re.search(r"Propose (\d+) novel", prompt).group(1))
            return json.dumps(
                [{"title": f"{proj} idea {k}", "text": f"{proj} hypothesis {k} distinct."} for k in range(n)]
            )
        if "Hold a lab meeting" in prompt:
            ids = re.findall(r"^\[(\w+)\]", prompt, re.M)
            return json.dumps({"feedback": [{"id": i, "feedback": f"fix {i}"} for i in ids], "lab_note": "use 1263"})
        if "Revise it" in prompt:
            title = re.search(r"HYPOTHESIS: (.+)", prompt).group(1)
            return json.dumps({"title": title + " (revised)", "text": "revised text"})
        if "Decide which of your group" in prompt:
            ids = re.findall(r"^\[(\w+)\]", prompt, re.M)
            return json.dumps({"ranking": ids[::-1], "reason": "r"})
        if "panel of expert reviewers" in prompt:
            return json.dumps({"winner": "A", "reason": "A", "against_B": "B weak"})
        return "{}"

    return call, calls


def run(monkeypatch, **kw):
    call, calls = fake_llm()
    monkeypatch.setattr(swarm, "call_llm", call)
    monkeypatch.setattr(swarm, "similarity_score", lambda a, b: 0.1)
    s = rg.ResearchGroupCoScientist("Find hypotheses.", model="m", parallel=1, seed=1, **kw)
    return s.run(), calls


def test_two_pis_four_students(monkeypatch):
    res, calls = run(monkeypatch, n_pis=2, n_students=4, per_student=2, pi_model="big", student_model="small")
    assert res["groups"] == {"1": [1, 2], "2": [3, 4]}
    hs = {h["id"]: h for h in res["hypotheses"]}
    final = [hs[i] for i in res["final"]]
    assert len(final) == 8 and all(h["parent_ids"] for h in final)  # every idea was revised once
    assert sum(h["forwarded"] for h in final) == 4  # each PI puts forward half its group's ideas
    assert all(h["forwarded"] for h in final[:4]) and not any(h["forwarded"] for h in final[4:])
    elos = [h["elo"] for h in final]
    assert elos == sorted(elos, reverse=True)  # the scorer orders by Elo, so Elo must follow the final ranking
    # PI calls go to the PI model, student calls to the student model
    for model, prompt in calls:
        if "You are a student" in prompt or "Revise it" in prompt:
            assert model == "small"
        else:
            assert model == "big"
    # with two PIs, a cross-group match is judged by one of them; every match records its PI
    assert res["matches"] and all(m["judge_pi"] in (1, 2) for m in res["matches"])
    assert not res["errors"]


def test_second_pi_sees_first_focus(monkeypatch):
    _, calls = run(monkeypatch, n_pis=2, n_students=2, per_student=1)
    agendas = [p for _, p in calls if "agenda: choose a focus" in p]
    assert "focus 1" not in agendas[0] and "PI 1: focus 1" in agendas[1]


def test_judge_is_from_neither_group_when_possible(monkeypatch):
    res, _ = run(monkeypatch, n_pis=4, n_students=4, per_student=2)
    for m in res["matches"]:
        assert m["judge_pi"] not in (m["winner_pi"], m["loser_pi"])


def test_one_pi_one_student(monkeypatch):
    res, _ = run(monkeypatch, n_pis=1, n_students=1, per_student=8)
    assert len(res["final"]) == 8 and len(res["decisions"]) == 1


def test_four_pis_one_student(monkeypatch):
    # co-advising: one student gets a project from each of four PIs, splits its 8 proposals between them,
    # reads the other advisors' briefings and hears every advisor's lab note when revising
    res, calls = run(monkeypatch, n_pis=4, n_students=1, per_student=8)
    assert res["groups"] == {"1": [1], "2": [1], "3": [1], "4": [1]}
    props = [p for _, p in calls if "Your project:" in p and "Propose" in p]
    assert len(props) == 4 and all("Propose 2 novel" in p and "co-advised by PIs 1, 2, 3, 4" in p for p in props)
    assert all(p.count("BRIEFING FROM PI") == 3 for p in props)
    revs = [p for _, p in calls if "Revise it" in p]
    assert len(revs) == 8 and all(p.count("use 1263") == 4 for p in revs)
    hs = {h["id"]: h for h in res["hypotheses"]}
    assert sorted(hs[i]["pi"] for i in res["final"]) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert sum(hs[i]["forwarded"] for i in res["final"]) == 4  # each PI puts forward one of the two ideas made for it
    assert all(m["judge_pi"] not in (m["winner_pi"], m["loser_pi"]) for m in res["matches"])
    assert not res["errors"]


def test_bad_split_rejected():
    with pytest.raises(ValueError):
        rg.ResearchGroupCoScientist("g", n_pis=3, n_students=4)
