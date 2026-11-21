"""Offline tests of peer-review mode (app/peer_review.py): the LLM boundary is mocked, no key and no network."""

import json
import re

import pytest

import app.peer_review as pr
import app.swarm as swarm


def fake_llm(score=None, drop_revision=None):
    calls = []

    def call(prompt, temperature=0.7, model=None):
        calls.append((model, prompt))
        if "orchestrator of a swarm" in prompt:
            n = int(re.search(r"into (\d+) distinct", prompt).group(1))
            return json.dumps(
                [{"name": f"D{i}", "objective": "o", "data_types": "t", "boundaries": "b"} for i in range(n)]
            )
        if "submitting hypotheses for peer review" in prompt:
            a = re.search(r"You are author (\d+)", prompt).group(1)
            n = int(re.search(r"Propose (\d+) novel", prompt).group(1))
            return json.dumps([{"title": f"A{a} idea {k}", "text": f"A{a} hypothesis {k}."} for k in range(n)])
        if 'Return only a JSON object: {"reviews"' in prompt:
            r = int(re.search(r"You are reviewer (\d+)", prompt).group(1))
            ids = re.findall(r"^\[(\w+)\]", prompt, re.M)
            rnd2 = "YOUR FIRST REVIEW" in prompt
            return json.dumps(
                {
                    "reviews": [
                        {
                            "id": i,
                            "score": score(i, r, rnd2) if score else 5,
                            "recommendation": "minor revision",
                            "comments": f"r{r} on {i}",
                        }
                        for i in ids
                    ]
                }
            )
        if "The reviewers have refereed your submission" in prompt:
            title = re.search(r"SUBMISSION: (.+)", prompt).group(1)
            if drop_revision and drop_revision in title:
                return "not json"
            return json.dumps({"title": title + " (revised)", "text": "revised", "response": "fixed"})
        return "{}"

    return call, calls


def run(monkeypatch, call, **kw):
    monkeypatch.setattr(swarm, "call_llm", call)
    monkeypatch.setattr(swarm, "similarity_score", lambda a, b: 0.1)
    return pr.PeerReviewCoScientist("Find hypotheses.", model="m", parallel=1, seed=1, **kw).run()


def test_four_authors_three_reviewers(monkeypatch):
    call, calls = fake_llm()
    res = run(monkeypatch, call, n_authors=4, n_reviewers=3, per_author=2, author_model="a", reviewer_model="r")
    hs = {h["id"]: h for h in res["hypotheses"]}
    final = [hs[i] for i in res["final"]]
    assert len(final) == 8 and all(h["parent_ids"] and len(h["reviews"]) == 3 for h in final)
    assert all(len(hs[h["parent_ids"][0]]["reviews"]) == 3 for h in final)  # every reviewer refereed both versions
    # one call per reviewer per round, one revision per submission
    roles = [c for c in calls if "You are reviewer" in c[1]]
    assert len(roles) == 6 and all(m == "r" for m, _ in roles)
    assert all(m == "a" for m, p in calls if "You are author" in p)
    assert res["settings"]["n_authors"] == 4 and res["settings"]["n_reviewers"] == 3
    assert not res["errors"]


def test_reviewers_are_blind_and_see_their_first_review(monkeypatch):
    call, calls = fake_llm()
    run(monkeypatch, call, n_authors=2, n_reviewers=2, per_author=2)
    r1 = [p for _, p in calls if "You are reviewer" in p and "YOUR FIRST REVIEW" not in p]
    r2 = [p for _, p in calls if "YOUR FIRST REVIEW" in p]
    assert len(r1) == 2 and len(r2) == 2
    assert not any(re.search(r"\(author \d\)|author 1|author 2", p.split("SUBMISSIONS:")[1]) for p in r1)
    assert all("r2 on" in p and "r1 on" not in p for p in r2 if "You are reviewer 2" in p)
    assert all("AUTHORS' RESPONSE: fixed" in p for p in r2)


def test_final_ranking_follows_second_scores(monkeypatch):
    # second-round score: author 2's ideas 9, author 1's 3; Elo must follow so the scorer keeps the order
    call, _ = fake_llm()

    def scored(prompt, temperature=0.7, model=None):
        out = call(prompt, temperature, model)
        if "YOUR FIRST REVIEW" in prompt:
            d = json.loads(out)
            titles = dict(re.findall(r"^\[(\w+)\] (A\d)", prompt, re.M))
            for v in d["reviews"]:
                v["score"] = 9 if titles[v["id"]] == "A2" else 3
            return json.dumps(d)
        return out

    res = run(monkeypatch, scored, n_authors=2, n_reviewers=1, per_author=2)
    hs = {h["id"]: h for h in res["hypotheses"]}
    final = [hs[i] for i in res["final"]]
    assert [h["author"] for h in final] == [2, 2, 1, 1]
    assert [h["score2"] for h in final] == [9, 9, 3, 3] and all(h["score1"] == 5 for h in final)
    elos = [h["elo"] for h in final]
    assert elos == sorted(elos, reverse=True)


def test_failed_revision_keeps_first_version(monkeypatch):
    call, _ = fake_llm(drop_revision="A1 idea 0")
    res = run(monkeypatch, call, n_authors=1, n_reviewers=1, per_author=3)
    hs = {h["id"]: h for h in res["hypotheses"]}
    final = [hs[i] for i in res["final"]]
    assert len(final) == 3
    kept = [h for h in final if not h["parent_ids"]]
    assert len(kept) == 1 and kept[0]["title"] == "A1 idea 0" and kept[0]["score"] == 5
    assert any("no parsable revision" in e for e in res["errors"])


def test_score_parsing():
    assert pr.PeerReviewCoScientist.score_of("8/10") == 8
    assert pr.PeerReviewCoScientist.score_of(7.5) == 7.5
    assert pr.PeerReviewCoScientist.score_of("high") is None
    assert pr.PeerReviewCoScientist.score_of(11) is None


def test_bad_settings_rejected():
    with pytest.raises(ValueError):
        pr.PeerReviewCoScientist("g", n_authors=2, n_reviewers=0)
    with pytest.raises(ValueError):
        pr.PeerReviewCoScientist("g", n_authors=0, n_reviewers=1)
