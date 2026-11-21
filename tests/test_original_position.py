"""Offline tests of original-position mode (app/original_position.py): the LLM boundary is mocked, no key and no network."""

import json
import re

import pytest

import app.original_position as op
import app.swarm as swarm


def fake_llm(score=None):
    calls = []

    def call(prompt, temperature=0.7, model=None):
        calls.append(prompt)
        if "orchestrator of a swarm" in prompt:
            n = int(re.search(r"into (\d+) distinct", prompt).group(1))
            return json.dumps(
                [{"name": f"D{i}", "objective": "o", "data_types": "t", "boundaries": "b"} for i in range(n)]
            )
        if "Merge them into one set" in prompt:
            return json.dumps({"principles": ["Quote only numbers in the files.", "State a test that could fail."]})
        if '{"principles"' in prompt:
            m = re.search(r"member (\d+)", prompt)
            return json.dumps({"principles": [f"principle from {'member ' + m.group(1) if m else 'behind the veil'}"]})
        if "Propose" in prompt and "novel, testable hypotheses" in prompt:
            m = re.search(r"You are member (\d+)", prompt).group(1)
            n = int(re.search(r"Propose (\d+) novel", prompt).group(1))
            return json.dumps([{"title": f"M{m} idea {k}", "text": f"M{m} hypothesis {k}."} for k in range(n)])
        if '{"scores"' in prompt:
            judge = int(re.search(r"You are member (\d+)", prompt).group(1))
            items = re.findall(r"^\[(\w+)\](?: \(YOUR OWN\))? M(\d) idea", prompt, re.M)
            return json.dumps(
                {"scores": [{"id": i, "score": score(judge, int(a)) if score else 6, "reason": "r"} for i, a in items]}
            )
        return "{}"

    return call, calls


def run(monkeypatch, call, **kw):
    monkeypatch.setattr(swarm, "call_llm", call)
    monkeypatch.setattr(swarm, "similarity_score", lambda a, b: 0.1)
    return op.OriginalPositionCoScientist("Find hypotheses.", model="m", parallel=1, seed=1, **kw).run()


def test_behind_the_veil(monkeypatch):
    call, calls = fake_llm()
    res = run(monkeypatch, call, n_members=3, per_member=2)
    # principles are drafted before any direction is assigned, without a member's direction or number
    first_orch = next(i for i, p in enumerate(calls) if "orchestrator of a swarm" in p)
    drafts = [i for i, p in enumerate(calls) if '{"principles"' in p and "Merge them" not in p]
    assert len(drafts) == 3 and max(drafts) < first_orch
    assert all("veil of ignorance" in calls[i] and "Your research direction" not in calls[i] for i in drafts)
    assert res["principles"] == ["Quote only numbers in the files.", "State a test that could fail."]
    judges = [p for p in calls if '{"scores"' in p]
    assert len(judges) == 3 and not any("(YOUR OWN)" in p for p in judges)
    assert all("Quote only numbers in the files." in p for p in calls if "novel, testable hypotheses" in p)
    assert res["settings"]["veil"] is True and res["settings"]["aggregate"] == "maximin"
    assert len(res["final"]) == 6 and not res["errors"]


def test_veil_lifted_marks_own_ideas(monkeypatch):
    call, calls = fake_llm()
    run(monkeypatch, call, n_members=2, per_member=2, veil=False)
    drafts = [p for p in calls if '{"principles"' in p and "Merge them" not in p]
    assert all("Your research direction is D" in p for p in drafts)
    for p in (p for p in calls if '{"scores"' in p):
        me = re.search(r"You are member (\d+)", p).group(1)
        marked = re.findall(r"^\[\w+\] \(YOUR OWN\) M(\d)", p, re.M)
        assert marked and set(marked) == {me}


def test_maximin_ranks_by_the_worst_score_and_self_bias(monkeypatch):
    # judge 1 loves its own ideas (10) and scores member 2's ideas 5; judge 2 scores everything 7.
    # Means: member 1's ideas 8.5, member 2's 6. Minimums: member 1's 7, member 2's 5 -> same order by both rules here,
    # so make member 2's ideas steadier: judge 2 gives member 1's ideas 3 -> member 1: min 3, mean 6.5; member 2: min 5, mean 6
    def score(judge, author):
        if judge == 1:
            return 10 if author == 1 else 5
        return 3 if author == 1 else 7

    call, _ = fake_llm(score)
    res = run(monkeypatch, call, n_members=2, per_member=2, veil=False)
    hs = {h["id"]: h for h in res["hypotheses"]}
    final = [hs[i] for i in res["final"]]
    assert [h["author"] for h in final] == [2, 2, 1, 1]  # maximin: member 2's ideas (worst 5) beat member 1's (worst 3)
    assert all(h["rank_mean"] in (1, 2) for h in final if h["author"] == 1)  # by the mean, member 1's ideas would lead
    j = {x["member"]: x for x in res["judgments"]}
    assert j[1]["self_bias"] == 5.0 and j[2]["self_bias"] == 4.0
    assert res["self_bias"] == 4.5
    elos = [h["elo"] for h in final]
    assert elos == sorted(elos, reverse=True)
    res2 = run(monkeypatch, call, n_members=2, per_member=2, veil=False, aggregate="mean")
    hs2 = {h["id"]: h for h in res2["hypotheses"]}
    assert [hs2[i]["author"] for i in res2["final"]] == [1, 1, 2, 2]


def test_settings_are_checked():
    with pytest.raises(ValueError):
        op.OriginalPositionCoScientist("g", n_members=0)
    with pytest.raises(ValueError):
        op.OriginalPositionCoScientist("g", aggregate="median")
    assert op.OriginalPositionCoScientist.score_of("8/10") == 8 and op.OriginalPositionCoScientist.score_of(0) is None
