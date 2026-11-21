"""Offline tests of the team modes in the Gradio app (Research group, Startup team, Peer review): no key, no network."""

import importlib.util
import json
import os
import re
import zipfile

import pytest

import app.swarm as swarm
from app.peer_review import PeerReviewCoScientist
from app.research_group import ResearchGroupCoScientist


@pytest.fixture(scope="module")
def app_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("gradio_app_team", os.path.join(root, "app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_group_team(app_module):
    team, org = app_module.build_team(
        "Research group",
        "goal",
        "m",
        lab_pis=2,
        lab_students=4,
        lab_total=8,
        lab_pi_model="big",
        lab_student_model=None,
    )
    assert isinstance(team, ResearchGroupCoScientist) and team.framing == "lab"
    assert (team.n_pis, team.n_workers, team.per_worker) == (2, 4, 2)
    assert (team.pi_model, team.student_model) == ("big", "m")
    assert org == "2 PIs, 4 students"


def test_startup_team_allows_cofounders_only(app_module):
    team, org = app_module.build_team("Startup team", "goal", "m", biz_founders=4, biz_juniors=0, biz_total=8)
    assert team.framing == "startup" and team.n_workers == 0 and team.per_worker == 2
    assert org == "4 co-founders (no junior members)"
    team, org = app_module.build_team("Startup team", "goal", "m", biz_founders=4, biz_juniors=1)
    assert org == "4 co-founders, 1 junior member"


def test_peer_review_team(app_module):
    team, org = app_module.build_team(
        "Peer review", "goal", "m", rev_authors=8, rev_reviewers=1, rev_total=8, rev_reviewer_model="Same as LLM Model"
    )
    assert isinstance(team, PeerReviewCoScientist)
    assert (team.n_authors, team.n_reviewers, team.per_worker, team.reviewer_model) == (8, 1, 1, "m")
    assert org == "8 authors, 1 reviewer, two review rounds"


def test_original_position_team(app_module):
    team, org = app_module.build_team("Original position", "goal", "m", veil_members=4, veil_total=8)
    assert team.veil is True and team.aggregate == "maximin" and (team.n_members, team.per_worker) == (4, 2)
    assert org == "4 members, behind the veil of ignorance, maximin ranking"
    team, org = app_module.build_team(
        "Original position", "goal", "m", veil_members=2, veil_on=False, veil_rank="Mean score"
    )
    assert team.veil is False and team.aggregate == "mean"
    assert org == "2 members, veil lifted, mean-score ranking"


def test_invalid_teams_are_explained(app_module):
    with pytest.raises(ValueError, match="divide one another"):
        app_module.build_team("Research group", "goal", "m", lab_pis=2, lab_students=3)
    with pytest.raises(ValueError, match="at least one student"):
        app_module.build_team("Research group", "goal", "m", lab_pis=1, lab_students=0)
    with pytest.raises(ValueError, match="divide one another"):
        app_module.build_team("Startup team", "goal", "m", biz_founders=4, biz_juniors=3)


def test_only_the_chosen_panel_is_shown(app_module):
    shown = [u["visible"] for u in app_module.mode_panels("Peer review")]
    assert shown == [False, False, False, True, False]
    assert [u["visible"] for u in app_module.mode_panels("Original position")][-1] is True
    assert [u["visible"] for u in app_module.mode_panels("Single co-scientist")] == [False] * 5


def fake_review_llm(prompt, temperature=0.7, model=None, fallback_models=None):
    if "orchestrator of a swarm" in prompt:
        n = int(re.search(r"into (\d+) distinct", prompt).group(1))
        return json.dumps([{"name": f"D{i}", "objective": "o", "data_types": "t", "boundaries": "b"} for i in range(n)])
    if "submitting hypotheses for peer review" in prompt:
        a = re.search(r"You are author (\d+)", prompt).group(1)
        n = int(re.search(r"Propose (\d+) novel", prompt).group(1))
        return json.dumps([{"title": f"A{a} idea {k}", "text": f"A{a} hypothesis {k}."} for k in range(n)])
    if '"reviews"' in prompt:
        ids = re.findall(r"^\[(\w+)\]", prompt, re.M)
        return json.dumps(
            {"reviews": [{"id": i, "score": 7, "recommendation": "accept", "comments": "c"} for i in ids]}
        )
    if "The reviewers have refereed your submission" in prompt:
        title = re.search(r"SUBMISSION: (.+)", prompt).group(1)
        return json.dumps({"title": title + " (revised)", "text": "revised", "response": "done"})
    return "{}"


def test_peer_review_run_streams_saves_and_exports(app_module, monkeypatch, tmp_path):
    monkeypatch.setattr(swarm, "call_llm", fake_review_llm)
    monkeypatch.setattr(swarm, "similarity_score", lambda a, b: 0.1)
    monkeypatch.setattr(app_module, "get_reports_dir", lambda: tmp_path)
    monkeypatch.setattr(app_module, "save_run", lambda **kw: {"run_id": "r1"})
    monkeypatch.setattr(app_module, "write_report", lambda saved: tmp_path / "r.html")
    log = []
    team, org = app_module.build_team(
        "Peer review", "Find hypotheses.", "m", rev_authors=2, rev_reviewers=1, rev_total=4, progress=log.append
    )
    steps = list(
        app_module.stream_swarm_run(team, "Find hypotheses.", "m", log, 60, 0.01, label="Peer review", organization=org)
    )
    status, html_out, _ = steps[-1]
    assert status.startswith("✅ Peer review finished: 4 hypotheses ranked") and org in status
    assert "<b>Peer review:</b> 2 authors, 1 reviewer" in html_out
    assert app_module.LAST_SWARM["mode"] == "peer review"
    path = app_module.export_bundle_zip("Swarm")
    assert path and zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as z:
        text = "".join(z.read(n).decode(errors="replace") for n in z.namelist() if n.endswith(".md"))
    assert "Mode: peer review" in text
