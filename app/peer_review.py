"""Peer-review mode: a swarm organized like a journal, with authors who submit and reviewers who referee.

Swarm mode (app.swarm) ranks ideas by pairwise matches. Peer review scores each submission on its own: an orchestrator
splits the goal into one direction per author; each author submits hypotheses in its direction; every reviewer referees
every submission blind (no author named), with a score from 1 to 10, a recommendation and comments, and checks the numbers
against the attached files; each author revises every submission and writes a response to the reviewers; the reviewers
score the revised submissions again, seeing their own first review and the response; and the final ranking orders the
submissions by the mean of the second scores (ties by the mean first score).

    n_authors    authors (directions); each submits per_author hypotheses
    n_reviewers  reviewers, each refereeing every submission in one call per review round
    author_model, reviewer_model  the model behind each role (default: model)

Every call is recorded with its role (author2, reviewer1 round 1, author2 revision, ...) and the model asked, and each
submission keeps its reviews, both rounds of scores, the response and the revision, so agreement between reviewers and the
change in score after revision can be counted.
"""

import statistics
import time
from typing import Any, Dict, List, Optional

from . import swarm as sw
from .swarm import SwarmCoScientist, as_text, reply_snippet

RECOMMENDATIONS = ("accept", "minor revision", "major revision", "reject")


class PeerReviewCoScientist(SwarmCoScientist):
    def __init__(
        self,
        goal: str,
        model: Optional[str] = None,
        n_authors: int = 4,
        n_reviewers: int = 3,
        per_author: int = 2,
        author_model: Optional[str] = None,
        reviewer_model: Optional[str] = None,
        final_k: int = 9,
        **kw,
    ):
        if n_authors < 1 or n_reviewers < 1 or per_author < 1:
            raise ValueError("peer review needs at least one author, one reviewer and one submission per author")
        for x in ("n_workers", "per_worker", "rounds", "matches_per_round"):
            kw.pop(x, None)
        super().__init__(
            goal,
            model=model,
            n_workers=n_authors,
            per_worker=per_author,
            rounds=1,
            matches_per_round=0,
            final_k=final_k,
            **kw,
        )
        self.n_reviewers = n_reviewers
        self.author_model, self.reviewer_model = author_model or model, reviewer_model or model
        self.reviews: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ plumbing
    def llm(self, role: str, prompt: str, temperature: float) -> str:
        """The swarm's llm(), with author calls sent to the author model and reviewer calls to the reviewer model."""
        model = (
            self.author_model
            if role.startswith("author")
            else self.reviewer_model
            if role.startswith("reviewer")
            else self.model
        )
        t0 = time.time()
        out = sw.call_llm(prompt, temperature=temperature, model=model)
        err = (out or "").startswith("Error:")
        self.trace.append(
            dict(
                role=role,
                model_requested=model,
                model_used=sw.last_model_used() if not err else None,
                seconds=round(time.time() - t0, 1),
                prompt_chars=len(prompt),
                response_chars=len(out or ""),
                error=err,
                prompt=prompt,
                response=out,
                **sw.last_usage(),
            )
        )
        if err:
            self.errors.append(f"{role}: {out[:300]}")
        return out or ""

    @staticmethod
    def score_of(v: Any) -> Optional[float]:
        try:
            x = float(str(v).split("/")[0].strip())
        except (TypeError, ValueError):
            return None
        return x if 1 <= x <= 10 else None

    # ------------------------------------------------------------------ routines
    def submit(self, args) -> List[Dict[str, Any]]:
        """One author's submissions: the swarm worker's prompt, framed as an author submitting for review."""
        w, lens = args
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            f"You are author {w + 1} of {self.n_authors}, submitting hypotheses for peer review. Your direction: "
            f"{lens.get('name', '')}.\nObjective: {lens.get('objective', '')}\nFavor data types: {lens.get('data_types', '')}\n"
            f"Boundaries (left to the other authors): {lens.get('boundaries', '')}\n\n"
            f"Propose {self.per_worker} novel, testable hypotheses within your direction, each complete in the format the goal "
            f"asks for. {self.n_reviewers} reviewer(s) will score each one, check every number against the attached files, "
            "and ask for revisions.\n\n"
            + self.data_block(
                f"{self.goal[:3000]} {lens.get('name', '')} {lens.get('objective', '')} {lens.get('data_types', '')}"
            )
            + "Please return the response as a JSON array of objects, where each object has a 'title' and 'text' key."
        )
        data, reply = self.ask_json(f"author{w + 1}", prompt, self.gen_t, list)
        out = []
        for d in data or []:
            if isinstance(d, dict) and d.get("title") and d.get("text"):
                out.append(
                    dict(
                        id=self.new_id("S"),
                        title=as_text(d["title"]),
                        text=as_text(d["text"]),
                        worker=w + 1,
                        author=w + 1,
                        lens=lens.get("name", ""),
                        round=1,
                        parent_ids=[],
                        elo=1200.0,
                        wins=0,
                        losses=0,
                        review_comments=[],
                        references=[],
                        reviews={},
                        active=True,
                    )
                )
        if not out:
            self.errors.append(f"author{w + 1}: no parsable hypotheses ({reply_snippet(reply)})")
        return out

    def referee(self, args) -> Dict[str, Dict[str, Any]]:
        """Reviewer r scores every submission in one call (round 1), or every revision (round 2). Returns {id: review}."""
        r, rnd, hs = args
        crit = (
            "fit to the goal; whether every number it quotes matches the attached files (name any that do not); whether its "
            "design and comparison can test the claim, and can tell it from the alternatives; novelty; expected impact."
        )
        if rnd == 1:
            task = (
                f"You are reviewer {r + 1} of {self.n_reviewers} on a panel refereeing hypotheses submitted for the research "
                "goal below. The authors are not named. Referee every submission on its own, against these criteria: "
                f"{crit} For each give a score from 1 (unsound) to 10 (outstanding), a recommendation ("
                + ", ".join(RECOMMENDATIONS)
                + "), and comments the author can act on.\n\n"
            )
            subs = "\n\n".join(f"[{h['id']}] {h['title']}\n{h['text'][:3000]}" for h in hs)
        else:
            task = (
                f"You are reviewer {r + 1} of {self.n_reviewers}. The authors have revised the submissions you refereed and "
                "replied to the reviews. For each revised submission, read your first review and the authors' response, judge "
                f"against the same criteria ({crit}) whether the revision fixed what was wrong, and give a new score from 1 to "
                "10, a recommendation (" + ", ".join(RECOMMENDATIONS) + ") and brief comments.\n\n"
            )
            subs = "\n\n".join(
                f"[{h['id']}] {h['title']}\n{h['text'][:3000]}\n"
                f"YOUR FIRST REVIEW (score {self.hyps[h['parent_ids'][0]]['reviews'].get(r + 1, {}).get('score', '?')}): "
                f"{self.hyps[h['parent_ids'][0]]['reviews'].get(r + 1, {}).get('comments', '(none)')[:1200]}\n"
                f"AUTHORS' RESPONSE: {h.get('response', '')[:1200] or '(none)'}"
                for h in hs
            )
        prompt = (
            task
            + f"RESEARCH GOAL (summary):\n{self.goal_brief(4000)}\n\nSUBMISSIONS:\n{subs}\n\n"
            + self.data_block(" ".join(f"{h['title']} {h['text'][:1500]}" for h in hs), min(self.data_budget, 10000))
            + 'Return only a JSON object: {"reviews": [{"id": "...", "score": 1-10, "recommendation": "...", "comments": "..."}]}'
        )
        role = f"reviewer{r + 1} round {rnd}"
        d, reply = self.ask_json(role, prompt, self.judge_t, dict)
        ids = {h["id"] for h in hs}
        out = {}
        for x in (d or {}).get("reviews") or []:
            if not isinstance(x, dict):
                continue
            i = as_text(x.get("id", "")).strip("[] ")
            s = self.score_of(x.get("score"))
            if i in ids and s is not None:
                rec = as_text(x.get("recommendation", "")).strip().lower()
                out[i] = dict(
                    reviewer=r + 1,
                    round=rnd,
                    score=s,
                    recommendation=rec if rec in RECOMMENDATIONS else rec[:40],
                    comments=as_text(x.get("comments", "")),
                )
        missing = [h["id"] for h in hs if h["id"] not in out]
        if missing:
            self.errors.append(f"{role}: no usable review of {', '.join(missing)} ({reply_snippet(reply)})")
        self.reviews.append(dict(reviewer=r + 1, round=rnd, n_submissions=len(hs), n_reviewed=len(out)))
        return out

    def revise(self, h: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        role = f"author{h['author']} revision"
        revs = [h["reviews"][r] for r in sorted(h["reviews"])]
        block = (
            "\n\n".join(
                f"REVIEWER {v['reviewer']} (score {v['score']:g}, {v['recommendation']}): {v['comments'][:2000]}"
                for v in revs
            )
            or "(no usable reviews)"
        )
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            f"You are author {h['author']}. The reviewers have refereed your submission below. Revise it: keep its core idea "
            "unless the reviews show it is wrong, address every point the reviewers raise, correct any number that the "
            "attached files do not support (say 'not in the files' rather than invent a number), and keep the full format "
            "the goal asks for. Then write a short response to the reviewers saying what you changed and, where you disagree, "
            "why.\n\n"
            f"SUBMISSION: {h['title']}\n{h['text']}\n\nREVIEWS:\n{block}\n\n"
            + self.data_block(f"{h['title']} {h['text'][:4000]}", min(self.data_budget, 12000))
            + "Return only a JSON object with keys 'title', 'text' and 'response'."
        )
        d, reply = self.ask_json(role, prompt, 0.6, dict)
        if not d or not d.get("text"):
            self.errors.append(f"{role} of {h['id']}: no parsable revision ({reply_snippet(reply)})")
            return None
        return dict(
            h,
            id=self.new_id("V"),
            title=as_text(d.get("title") or h["title"]),
            text=as_text(d["text"]),
            response=as_text(d.get("response", "")),
            round=2,
            parent_ids=[h["id"]],
            review_comments=[v["comments"] for v in revs],
            reviews={},
            active=True,
        )

    # ------------------------------------------------------------------ the run
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        self.progress("orchestrator: splitting the goal into one direction per author")
        lenses = self.orchestrate()
        self.progress(f"{len(lenses)} author(s) submitting in parallel")
        pool = [h for hs in self.pmap(self.submit, list(enumerate(lenses))) for h in hs]
        self.progress(f"proximity: de-duplicating {len(pool)} submissions")
        self.dedupe(pool)
        subs = [h for h in self.hyps.values() if h["active"]]
        final: List[Dict[str, Any]] = []
        if not subs:
            self.errors.append("no hypotheses were submitted: every author reply was empty or unreadable")
            self.progress("stopped: nothing to review")
        else:
            self.progress(f"round 1: {self.n_reviewers} reviewer(s) refereeing {len(subs)} submissions")
            for got in self.pmap(self.referee, [(r, 1, subs) for r in range(self.n_reviewers)]):
                for i, v in got.items():
                    self.hyps[i]["reviews"][v["reviewer"]] = v
            for h in subs:
                h["review_comments"] = [v["comments"] for v in h["reviews"].values()]
                h["score1"] = statistics.mean(v["score"] for v in h["reviews"].values()) if h["reviews"] else None
            self.progress(f"authors revising {len(subs)} submissions")
            current = []
            for old, new in zip(subs, self.pmap(self.revise, subs)):
                if new:
                    old["active"], old["revised_into"] = False, new["id"]
                    self.hyps[new["id"]] = new
                    current.append(new)
                else:
                    current.append(old)  # the first version stands, with its first-round scores
            revised = [h for h in current if h["parent_ids"]]
            if revised:
                self.progress(f"round 2: {self.n_reviewers} reviewer(s) scoring {len(revised)} revisions")
                for got in self.pmap(self.referee, [(r, 2, revised) for r in range(self.n_reviewers)]):
                    for i, v in got.items():
                        self.hyps[i]["reviews"][v["reviewer"]] = v
            for h in current:
                first = self.hyps[h["parent_ids"][0]] if h["parent_ids"] else h
                h["score1"] = first.get("score1")
                h["score2"] = (
                    statistics.mean(v["score"] for v in h["reviews"].values())
                    if h["parent_ids"] and h["reviews"]
                    else None
                )
                h["score"] = h["score2"] if h["score2"] is not None else h["score1"]
                if h["parent_ids"]:
                    h["review_comments"] = list(first["review_comments"]) + [
                        v["comments"] for v in h["reviews"].values()
                    ]
            final = sorted(
                current,
                key=lambda h: (
                    -(h["score"] if h["score"] is not None else 0),
                    -(h["score1"] if h["score1"] is not None else 0),
                    h["id"],
                ),
            )
            for n, h in enumerate(final):  # the scorer orders by Elo, so Elo follows the review ranking
                h["elo"], h["review_rank"] = 1200.0 - 10.0 * n, n + 1
        if self.data:
            for h in self.hyps.values():
                h["data_check"] = self.data.check_numbers(h["title"] + "\n" + h["text"])
        self.progress(f"done: {len(self.trace)} LLM calls in {time.time() - t0:.0f}s")
        return dict(
            settings=dict(
                model=self.model,
                mode="peer_review",
                n_authors=self.n_authors,
                n_reviewers=self.n_reviewers,
                n_workers=self.n_workers,
                per_worker=self.per_worker,
                author_model=self.author_model,
                reviewer_model=self.reviewer_model,
                rounds=1,
                review_rounds=2,
                matches_per_round=0,
                final_k=self.final_k,
                dedupe_threshold=self.dedupe_threshold,
                parallel=self.parallel,
                data=self.data.summary() if self.data else None,
                web=None,
            ),
            lenses=lenses,
            reviews=self.reviews,
            web_search=[],
            hypotheses=[dict(h, reviews=[h["reviews"][r] for r in sorted(h["reviews"])]) for h in self.hyps.values()],
            final=[h["id"] for h in final[: self.final_k]],
            merged=self.merged,
            matches=[],
            meta_reviews=[],
            errors=self.errors,
            seconds=round(time.time() - t0, 1),
            trace=self.trace,
        )

    @property
    def n_authors(self) -> int:
        return self.n_workers
