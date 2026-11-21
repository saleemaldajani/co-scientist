"""Original-position mode: a swarm that sets its standards and judges behind a veil of ignorance (Rawls, A Theory of Justice).

In Rawls's original position, the parties choose the principles of justice without knowing the place they will hold in
society, so no one can tailor the principles to their own advantage, and the principles chosen favor the worst-off position
(the difference principle, a maximin rule). This mode carries the idea into a swarm of co-scientists:

1. Principles behind the veil. Before members know their research direction or what they will propose, each writes the
   principles by which the team should judge hypotheses for the goal; one call merges them into a single short set.
2. Proposal. The orchestrator gives each member a direction, and each proposes hypotheses that the agreed principles will
   judge.
3. Judgment behind the veil. Every member scores every hypothesis against the principles (1 to 10) without authors named
   and without being told which hypotheses, if any, are its own.
4. Final ranking by maximin. A hypothesis ranks by its lowest score from any member (ties by the mean), so it must satisfy
   every member, as the difference principle favors the worst-off position; aggregate="mean" ranks by the mean instead.

With veil=False the veil is lifted, as a control: members write the principles after learning their direction and knowing
their hypotheses will be judged, and when judging they are told which hypotheses are their own. Each member's self-bias
(its mean score of its own hypotheses minus its mean score of the others') is recorded in both conditions.

    n_members  members (directions); each proposes per_member hypotheses and judges all of them
    veil       True (behind the veil) or False (veil lifted)
    aggregate  "maximin" (Rawls) or "mean"
"""

import statistics
import time
from typing import Any, Dict, List, Optional

from .swarm import SwarmCoScientist, as_text, reply_snippet

AGGREGATES = ("maximin", "mean")
MAX_PRINCIPLES = 6


class OriginalPositionCoScientist(SwarmCoScientist):
    def __init__(
        self,
        goal: str,
        model: Optional[str] = None,
        n_members: int = 4,
        per_member: int = 2,
        veil: bool = True,
        aggregate: str = "maximin",
        final_k: int = 9,
        **kw,
    ):
        if n_members < 1 or per_member < 1:
            raise ValueError("the original position needs at least one member and one hypothesis per member")
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}")
        for x in ("n_workers", "per_worker", "rounds", "matches_per_round"):
            kw.pop(x, None)
        super().__init__(
            goal,
            model=model,
            n_workers=n_members,
            per_worker=per_member,
            rounds=1,
            matches_per_round=0,
            final_k=final_k,
            **kw,
        )
        self.veil, self.aggregate = bool(veil), aggregate
        self.principles: List[str] = []
        self.principle_drafts: List[Dict[str, Any]] = []
        self.judgments: List[Dict[str, Any]] = []

    @staticmethod
    def score_of(v: Any) -> Optional[float]:
        try:
            x = float(str(v).split("/")[0].strip())
        except (TypeError, ValueError):
            return None
        return x if 1 <= x <= 10 else None

    # ------------------------------------------------------------------ routines
    def draft_principles(self, args) -> List[str]:
        """Member m's principles for judging hypotheses: behind the veil, or knowing its direction (veil lifted)."""
        m, lens = args
        if self.veil:
            position = (
                f"You are one of {self.n_members} members of a research team. You do not yet know which research direction "
                "you will be given, which hypotheses you will write, or whether any of the hypotheses later judged will be "
                "your own. From behind this veil of ignorance, propose the principles by which the team should judge "
                "hypotheses for the goal below: principles you would accept whichever position you end up in."
            )
        else:
            position = (
                f"You are member {m + 1} of {self.n_members} of a research team. Your research direction is "
                f"{lens.get('name', '')}: {lens.get('objective', '')}. You will propose hypotheses in this direction, and they "
                "will be judged with the other members' hypotheses. Propose the principles by which the team should judge "
                "hypotheses for the goal below."
            )
        prompt = (
            position
            + f"\n\nRESEARCH GOAL:\n{self.goal_brief(5000)}\n\n"
            + f"Give at most {MAX_PRINCIPLES - 1} principles, each one sentence that a judge can apply to a single hypothesis.\n"
            + 'Return only a JSON object: {"principles": ["...", "..."]}'
        )
        d, reply = self.ask_json(f"member{m + 1} principles", prompt, self.gen_t, dict)
        out = [as_text(p).strip() for p in (d or {}).get("principles") or [] if as_text(p).strip()]
        if not out:
            self.errors.append(f"member{m + 1} principles: none readable ({reply_snippet(reply)})")
        self.principle_drafts.append(dict(member=m + 1, principles=out))
        return out

    def merge_principles(self, drafts: List[List[str]]) -> List[str]:
        flat = [p for d in drafts for p in d]
        if not flat:
            return []
        listing = "\n".join(f"- {p}" for p in flat)
        prompt = (
            f"Members of a research team each proposed principles for judging hypotheses for the goal below"
            f"{', without knowing their own position in the team' if self.veil else ''}. Merge them into one set of at most "
            f"{MAX_PRINCIPLES} principles that every member could accept: keep what they share, combine near-duplicates, "
            "and drop any principle that would favor one kind of hypothesis or direction over another without reason.\n\n"
            f"RESEARCH GOAL (summary):\n{self.goal_brief(3000)}\n\nPROPOSED PRINCIPLES:\n{listing}\n\n"
            'Return only a JSON object: {"principles": ["...", "..."]}'
        )
        d, reply = self.ask_json("principles merge", prompt, 0.3, dict)
        out = [as_text(p).strip() for p in (d or {}).get("principles") or [] if as_text(p).strip()][:MAX_PRINCIPLES]
        if not out:
            self.errors.append(f"principles merge: fell back to the members' own list ({reply_snippet(reply)})")
            seen, out = set(), []
            for p in flat:
                if p.lower() not in seen:
                    seen.add(p.lower())
                    out.append(p)
            out = out[:MAX_PRINCIPLES]
        return out

    def propose(self, args) -> List[Dict[str, Any]]:
        m, lens = args
        rules = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(self.principles)) or "(none agreed)"
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            f"You are member {m + 1} of {self.n_members} of a research team. Your direction: {lens.get('name', '')}.\n"
            f"Objective: {lens.get('objective', '')}\nFavor data types: {lens.get('data_types', '')}\n"
            f"Boundaries (left to the other members): {lens.get('boundaries', '')}\n\n"
            f"The team agreed to judge every hypothesis by these principles:\n{rules}\n\n"
            f"Propose {self.per_worker} novel, testable hypotheses within your direction, each complete in the format the goal "
            "asks for.\n\n"
            + self.data_block(
                f"{self.goal[:3000]} {lens.get('name', '')} {lens.get('objective', '')} {lens.get('data_types', '')}"
            )
            + "Please return the response as a JSON array of objects, where each object has a 'title' and 'text' key."
        )
        data, reply = self.ask_json(f"member{m + 1}", prompt, self.gen_t, list)
        out = []
        for d in data or []:
            if isinstance(d, dict) and d.get("title") and d.get("text"):
                out.append(
                    dict(
                        id=self.new_id("H"),
                        title=as_text(d["title"]),
                        text=as_text(d["text"]),
                        worker=m + 1,
                        author=m + 1,
                        lens=lens.get("name", ""),
                        round=1,
                        parent_ids=[],
                        elo=1200.0,
                        wins=0,
                        losses=0,
                        review_comments=[],
                        references=[],
                        scores={},
                        active=True,
                    )
                )
        if not out:
            self.errors.append(f"member{m + 1}: no parsable hypotheses ({reply_snippet(reply)})")
        return out

    def judge_all(self, args) -> Dict[str, Dict[str, Any]]:
        """Member m scores every hypothesis against the principles in one call. Returns {id: score record}."""
        m, hs = args
        rules = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(self.principles)) or "(none agreed)"
        own = {h["id"] for h in hs if h["author"] == m + 1}
        if self.veil:
            position = (
                f"You are member {m + 1} of {self.n_members} of a research team, judging the team's hypotheses. Authors are "
                "not named, and you are not told which hypotheses, if any, are your own: judge each one as you would want "
                "yours judged."
            )
        else:
            position = (
                f"You are member {m + 1} of {self.n_members} of a research team, judging the team's hypotheses. Hypotheses "
                "marked (YOUR OWN) are the ones you wrote."
            )
        listing = "\n\n".join(
            f"[{h['id']}]{' (YOUR OWN)' if not self.veil and h['id'] in own else ''} {h['title']}\n{h['text'][:2500]}"
            for h in hs
        )
        ids = ", ".join(h["id"] for h in hs)
        prompt = (
            position
            + f"\n\nJudge every hypothesis on its own against the principles the team agreed:\n{rules}\n\n"
            + "Give each a score from 1 (fails the principles) to 10 (meets every principle) and one sentence of reasons.\n\n"
            + f"RESEARCH GOAL (summary):\n{self.goal_brief(4000)}\n\nHYPOTHESES:\n{listing}\n\n"
            + self.data_block(" ".join(f"{h['title']} {h['text'][:1500]}" for h in hs), min(self.data_budget, 10000))
            + f"Score each of these ids exactly once: {ids}.\n"
            + 'Return only a JSON object: {"scores": [{"id": "...", "score": 1-10, "reason": "..."}]}'
        )
        role = f"member{m + 1} judgment"
        d, reply = self.ask_json(role, prompt, self.judge_t, dict)
        valid = {h["id"] for h in hs}
        out = {}
        for x in (d or {}).get("scores") or []:
            if not isinstance(x, dict):
                continue
            i = as_text(x.get("id", "")).strip("[] ")
            s = self.score_of(x.get("score"))
            if i in valid and s is not None and i not in out:
                out[i] = dict(member=m + 1, score=s, own=i in own, reason=as_text(x.get("reason", "")))
        missing = [i for i in valid if i not in out]
        if missing:
            self.errors.append(f"{role}: no usable score for {', '.join(sorted(missing))} ({reply_snippet(reply)})")
        own_s = [v["score"] for v in out.values() if v["own"]]
        other_s = [v["score"] for v in out.values() if not v["own"]]
        self.judgments.append(
            dict(
                member=m + 1,
                n_scored=len(out),
                n_own=len(own_s),
                mean_own=round(statistics.mean(own_s), 2) if own_s else None,
                mean_others=round(statistics.mean(other_s), 2) if other_s else None,
                self_bias=round(statistics.mean(own_s) - statistics.mean(other_s), 2) if own_s and other_s else None,
            )
        )
        return out

    # ------------------------------------------------------------------ the run
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        if self.veil:
            self.progress(
                f"behind the veil: {self.n_members} member(s) drafting the principles before learning their position"
            )
            drafts = self.pmap(self.draft_principles, [(m, {}) for m in range(self.n_members)])
            self.principles = self.merge_principles(drafts)
            self.progress("orchestrator: splitting the goal into one direction per member")
            lenses = self.orchestrate()
        else:
            self.progress("orchestrator: splitting the goal into one direction per member (veil lifted)")
            lenses = self.orchestrate()
            self.progress(f"{self.n_members} member(s) drafting the principles, knowing their directions")
            drafts = self.pmap(self.draft_principles, list(enumerate(lenses)))
            self.principles = self.merge_principles(drafts)
        self.progress(f"{len(self.principles)} principles agreed; {len(lenses)} member(s) proposing in parallel")
        pool = [h for hs in self.pmap(self.propose, list(enumerate(lenses))) for h in hs]
        self.progress(f"proximity: de-duplicating {len(pool)} hypotheses")
        self.dedupe(pool)
        hs = [h for h in self.hyps.values() if h["active"]]
        final: List[Dict[str, Any]] = []
        if not hs:
            self.errors.append("no hypotheses were proposed: every member reply was empty or unreadable")
            self.progress("stopped: nothing to judge")
        else:
            veil = "behind the veil" if self.veil else "veil lifted"
            self.progress(f"judgment ({veil}): {self.n_members} member(s) scoring {len(hs)} hypotheses")
            for got in self.pmap(self.judge_all, [(m, hs) for m in range(self.n_members)]):
                for i, v in got.items():
                    self.hyps[i]["scores"][v["member"]] = v
            for h in hs:
                ss = [v["score"] for v in h["scores"].values()]
                h["min_score"] = min(ss) if ss else None
                h["mean_score"] = round(statistics.mean(ss), 3) if ss else None
                h["review_comments"] = [v["reason"] for v in h["scores"].values() if v["reason"]]

            def key(kind):
                def k(h):
                    lo, mu = h["min_score"] or 0, h["mean_score"] or 0
                    return (-lo, -mu, h["id"]) if kind == "maximin" else (-mu, -lo, h["id"])

                return k

            for kind in AGGREGATES:
                for n, h in enumerate(sorted(hs, key=key(kind))):
                    h[f"rank_{kind}"] = n + 1
            final = sorted(hs, key=key(self.aggregate))
            for n, h in enumerate(final):  # the scorer orders by Elo, so Elo follows the chosen ranking
                h["elo"] = 1200.0 - 10.0 * n
        if self.data:
            for h in self.hyps.values():
                h["data_check"] = self.data.check_numbers(h["title"] + "\n" + h["text"])
        bias = [j["self_bias"] for j in self.judgments if j["self_bias"] is not None]
        self.progress(f"done: {len(self.trace)} LLM calls in {time.time() - t0:.0f}s")
        return dict(
            settings=dict(
                model=self.model,
                mode="original_position",
                veil=self.veil,
                aggregate=self.aggregate,
                n_members=self.n_members,
                n_workers=self.n_workers,
                per_worker=self.per_worker,
                rounds=1,
                matches_per_round=0,
                final_k=self.final_k,
                dedupe_threshold=self.dedupe_threshold,
                parallel=self.parallel,
                data=self.data.summary() if self.data else None,
                web=None,
            ),
            lenses=lenses,
            principles=self.principles,
            principle_drafts=self.principle_drafts,
            judgments=sorted(self.judgments, key=lambda j: j["member"]),
            self_bias=round(statistics.mean(bias), 3) if bias else None,
            web_search=[],
            hypotheses=[dict(h, scores=[h["scores"][m] for m in sorted(h["scores"])]) for h in self.hyps.values()],
            final=[h["id"] for h in final[: self.final_k]],
            merged=self.merged,
            matches=[],
            meta_reviews=[],
            errors=self.errors,
            seconds=round(time.time() - t0, 1),
            trace=self.trace,
        )

    @property
    def n_members(self) -> int:
        return self.n_workers
