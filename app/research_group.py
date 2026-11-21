"""Research-group mode: a swarm organized like a laboratory, with principal investigators (PIs) and students.

Swarm mode (app.swarm) is flat: an orchestrator splits the goal, workers generate in parallel and a judge ranks every idea.
A research group is hierarchical. Each PI sets a focus for the group, gives every student a project and briefs the group
on the facts in the attached files; students propose and submit to their PI; the PI reviews the submissions at a lab
meeting, with feedback for each and one note for the whole group; students revise; the PI decides which of the group's
ideas to put forward; and the PIs compare the ideas put forward by all groups in pairwise matches (Elo, Swiss pairing,
the swarm's judge prompt), each match judged by a PI who leads neither idea's group when there is one.

    n_pis       number of groups (each led by one PI)
    n_students  students in all, split evenly between the PIs; with fewer students than PIs, each student is co-advised
                by n_pis / n_students PIs (one must divide the other), gets a project from each and splits its proposals
                between them, and hears every advisor's briefing and lab note
    per_student hypotheses each student proposes
                0 = PIs only: each PI proposes, another PI reviews (a PI alone reviews its own), the author revises
    per_student hypotheses each student proposes (each PI, with no students)
    pi_model, student_model  the model behind each role (default: model), so a large PI can lead small students
    framing     "lab" (PIs and students in research groups) or "startup" (co-founders, each leading a team of junior
                members): the same steps with the role words changed; the goal text says what kind of hypotheses to write

The final ranking lists the ideas put forward by Elo, then the others in their PI's order. Every call is recorded with
its role (pi1 agenda, student3, pi2 review, ...) and the model asked, and the result keeps each idea's student, PI,
feedback, revision and whether it was put forward, so the share of each group in the final ranking can be counted.
"""

import math
import re
import time
from typing import Any, Dict, List, Optional

from . import swarm as sw
from .swarm import DEFAULT_LENSES, SwarmCoScientist, as_text, expected, reply_snippet

# role words for framing="startup", applied to the instruction text only (never to the goal, the data or a model's text)
STARTUP_WORDS = [
    (r"the principal investigator of a research group", "a co-founder of a startup, leading a team"),
    (r"principal investigator", "co-founder"),
    (r"research groups?", "team"),
    (r"the department", "the startup"),
    (r"lab meeting", "team meeting"),
    (r"YOUR PI'S", "YOUR CO-FOUNDER'S"),
    (r"\bPIs\b", "co-founders"),
    (r"\bPI\b", "co-founder"),
    (r"\bstudents\b", "junior team members"),
    (r"\bstudent\b", "junior team member"),
    (r"co-advised", "jointly supervised"),
    (r"\badvisors\b", "supervisors"),
    (r"\bgroups\b", "teams"),
    (r"\bgroup\b", "team"),
]


class ResearchGroupCoScientist(SwarmCoScientist):
    def __init__(
        self,
        goal: str,
        model: Optional[str] = None,
        n_pis: int = 1,
        n_students: int = 4,
        per_student: int = 2,
        pi_model: Optional[str] = None,
        student_model: Optional[str] = None,
        forward_share: float = 0.5,
        matches_per_round: int = 2,
        final_k: int = 9,
        framing: str = "lab",
        **kw,
    ):
        if n_pis < 1 or n_students < 0 or (n_students and n_students % n_pis and n_pis % n_students):
            raise ValueError("n_students and n_pis must divide one another (or n_students = 0)")
        if framing not in ("lab", "startup"):
            raise ValueError("framing must be 'lab' or 'startup'")
        kw.pop("n_workers", None)
        kw.pop("per_worker", None)
        kw.pop("rounds", None)
        super().__init__(
            goal,
            model=model,
            n_workers=n_students,
            per_worker=per_student,
            rounds=1,
            matches_per_round=matches_per_round,
            final_k=final_k,
            **kw,
        )
        self.n_pis = n_pis
        self.pi_model, self.student_model = pi_model or model, student_model or model
        self.forward_share, self.framing = forward_share, framing
        if n_students == 0:  # PIs only
            self.groups = {p: [] for p in range(n_pis)}
        elif n_students >= n_pis:
            k = n_students // n_pis
            self.groups = {p: list(range(p * k, (p + 1) * k)) for p in range(n_pis)}  # PI index -> student indices
        else:  # co-advising: n_pis / n_students PIs share each student
            r = n_pis // n_students
            self.groups = {p: [p // r] for p in range(n_pis)}
        self.agendas: List[Dict[str, Any]] = []
        self.lab_meetings: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ plumbing
    def llm(self, role: str, prompt: str, temperature: float) -> str:
        """The swarm's llm(), with PI calls sent to the PI model and student calls to the student model."""
        model = (
            self.pi_model if role.startswith("pi") else self.student_model if role.startswith("student") else self.model
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

    def T(self, text: str) -> str:
        """Instruction text in the run's framing (unchanged for the lab)."""
        if self.framing == "startup":
            for a, b in STARTUP_WORDS:
                text = re.sub(a, b, text)
        return text

    def reviewer_of(self, p: int) -> int:
        """Who reviews group p's ideas: its PI, or with no students the next PI (a PI alone reviews its own)."""
        return p if self.n_workers else (p + 1) % self.n_pis

    def advisors(self, s: int) -> List[int]:
        return [p for p, ss in self.groups.items() if s in ss]

    def group_of(self, h: Dict[str, Any]) -> int:
        return h["pi"] - 1

    def submissions(self, hs: List[Dict[str, Any]], n: int = 3000) -> str:
        who = (
            (lambda h: self.T(f"(student {h['worker']})")) if self.n_workers else (lambda h: self.T(f"(PI {h['pi']})"))
        )
        return "\n\n".join(f"[{h['id']}] {who(h)} {h['title']}\n{h['text'][:n]}" for h in hs)

    # ------------------------------------------------------------------ routines
    def agenda(self, p: int, earlier: List[Dict[str, Any]]) -> Dict[str, Any]:
        students = self.groups[p]
        if earlier:
            others = (
                self.T(
                    "The other groups have already chosen these foci; choose a different one, so that together the groups "
                    "cover the goal:\n"
                )
                + "\n".join(self.T(f"- PI {a['pi']}: ") + a["focus"] for a in earlier)
                + "\n\n"
            )
        elif self.n_pis > 1:
            others = self.T(
                "Other groups will take other parts of the goal; choose one part, so that together the groups cover the goal.\n\n"
            )
        else:
            others = ""
        co = sorted({q + 1 for s in students for q in self.advisors(s) if q != p})
        if co:
            others += self.T(
                f"Your student is co-advised with PI(s) {', '.join(map(str, co))}, who will give the student other projects.\n\n"
            )
        if students:
            task = self.T(
                f"You are PI {p + 1} of {self.n_pis}, the principal investigator of a research group with {len(students)} "
                "student(s), working on the research goal below. As PI you set the group's agenda: choose a focus for the "
                "group, give each student a distinct project, and brief the group on what it must know before starting: the "
                "facts and numbers in the attached files that bear on the goal, the comparisons worth making, and the mistakes "
                "to avoid.\n\n"
            )
            fmt = (
                'Return only a JSON object: {"focus": "...", "briefing": "...", "projects": ['
                + f'{len(students)} object(s) with keys "name", "objective", "data_types", "boundaries"]}}'
            )
        else:  # PIs only: the PI will write the hypotheses; the briefing goes to the other PIs
            task = self.T(
                f"You are PI {p + 1} of {self.n_pis}, working on the research goal below without students: you will propose "
                "the hypotheses yourself. Choose a focus and write a briefing"
                + (" for the other PIs" if self.n_pis > 1 else " for yourself")
                + " on what bears on the goal: the facts and numbers in the attached files, the comparisons worth making, and "
                "the mistakes to avoid.\n\n"
            )
            fmt = 'Return only a JSON object: {"focus": "...", "briefing": "..."}'
        prompt = (
            task
            + others
            + f"RESEARCH GOAL:\n{self.goal_brief()}\n\n"
            + self.data_block(self.goal, min(self.data_budget, 8000))
            + fmt
        )
        d, reply = self.ask_json(f"pi{p + 1} agenda", prompt, 0.5, dict)
        d = d or {}
        if not students:
            focus = as_text(d.get("focus", ""))
            if not focus:
                self.errors.append(f"pi{p + 1} agenda: no focus ({reply_snippet(reply)})")
                focus = DEFAULT_LENSES[p % len(DEFAULT_LENSES)]["objective"]
            own = dict(name=focus[:120], objective=focus, data_types="", boundaries="")
            return dict(pi=p + 1, focus=focus, briefing=as_text(d.get("briefing", "")), projects=[own])
        projects = [x for x in (d.get("projects") or []) if isinstance(x, dict) and x.get("objective")][: len(students)]
        if len(projects) < len(students):
            self.errors.append(f"pi{p + 1} agenda: fell back to default projects ({reply_snippet(reply)})")
            pad = (DEFAULT_LENSES * (len(students) + p + 1))[p * len(students) :]
            projects = (projects + pad)[: len(students)]
        a = dict(
            pi=p + 1,
            focus=as_text(d.get("focus", "")) or projects[0].get("name", ""),
            briefing=as_text(d.get("briefing", "")),
            projects=[{k: as_text(v) for k, v in x.items()} for x in projects],
        )
        return a

    def propose(self, args) -> List[Dict[str, Any]]:
        s, p, project = args
        a = self.agendas[p]
        query = f"{self.goal[:3000]} {project.get('name', '')} {project.get('objective', '')} {project.get('data_types', '')}"
        if s is None:  # PIs only: the PI proposes, having read the other PIs' briefings
            n, role = self.per_worker, f"pi{p + 1} proposal"
            other = [o for o in self.agendas if o["pi"] != p + 1]
            body = (
                self.T(f"You are PI {p + 1} of {self.n_pis}, and you propose the hypotheses yourself. Your focus: ")
                + a["focus"]
                + "\n\n"
                + self.T("YOUR BRIEFING:\n")
                + a["briefing"][:3000]
                + "\n\n"
                + "".join(self.T(f"BRIEFING FROM PI {o['pi']}:\n") + o["briefing"][:1500] + "\n\n" for o in other)
                + self.T(
                    f"Propose {n} novel, testable hypotheses within your focus, each complete in the format the goal asks for.\n\n"
                )
            )
        else:
            adv = self.advisors(s)
            n, role = max(1, self.per_worker // len(adv)), f"student{s + 1}"
            where = f" and {self.n_workers} in the department" if self.n_pis > 1 else ""
            if len(adv) > 1:
                other = [self.agendas[q] for q in adv if q != p]
                who = (
                    self.T(
                        f"You are a student co-advised by PIs {', '.join(str(q + 1) for q in adv)}; this proposal is for the "
                        f"project PI {p + 1} gave you. PI {p + 1} set the focus: "
                    )
                    + a["focus"]
                    + "\n"
                )
                extra = "".join(
                    self.T(f"BRIEFING FROM PI {o['pi']} (another of your advisors):\n") + o["briefing"][:1500] + "\n\n"
                    for o in other
                )
            else:
                who = (
                    self.T(
                        f"You are a student in the research group of PI {p + 1}, one of {len(self.groups[p])} student(s) in "
                        f"the group{where}. Your PI set the group's focus: "
                    )
                    + a["focus"]
                    + "\n"
                )
                extra = ""
            body = (
                who + f"Your project: {project.get('name', '')}.\nObjective: {project.get('objective', '')}\n"
                f"Favor data types: {project.get('data_types', '')}\n"
                + self.T("Boundaries (left to the other students): ")
                + f"{project.get('boundaries', '')}\n\n"
                + self.T("YOUR PI'S BRIEFING:\n")
                + a["briefing"][:3000]
                + "\n\n"
                + extra
                + self.T(
                    f"Propose {n} novel, testable hypotheses within your project, each complete in the format the goal "
                    "asks for, to submit to your PI.\n\n"
                )
            )
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            + body
            + self.data_block(query)
            + "Please return the response as a JSON array of objects, where each object has a 'title' and 'text' key."
        )
        data, reply = self.ask_json(role, prompt, self.gen_t, list)
        out = []
        for d in data or []:
            if isinstance(d, dict) and d.get("title") and d.get("text"):
                out.append(
                    dict(
                        id=self.new_id("S"),
                        title=as_text(d["title"]),
                        text=as_text(d["text"]),
                        worker=0 if s is None else s + 1,
                        pi=p + 1,
                        lens=project.get("name", ""),
                        round=1,
                        parent_ids=[],
                        elo=1200.0,
                        wins=0,
                        losses=0,
                        review_comments=[],
                        references=[],
                        active=True,
                    )
                )
        if not out:
            self.errors.append(f"{role}: no parsable hypotheses ({reply_snippet(reply)})")
        return out

    def lab_meeting(self, p: int) -> str:
        hs = [h for h in self.hyps.values() if h["active"] and self.group_of(h) == p]
        if not hs:
            return ""
        r = self.reviewer_of(p)
        crit = (
            "whether it fits the goal, whether every number it quotes matches the attached files (name any that do not), "
            "whether its design and comparison test the claim, and what to change."
        )
        if self.n_workers:
            task = self.T(
                f"You are PI {p + 1}, the principal investigator of a research group. Your students have submitted the "
                "hypotheses below for the research goal. Hold a lab meeting: for each submission give specific feedback the "
                f"student can act on: {crit} Then write one note for the whole group: facts every student should use, "
                "mistakes several made, and overlaps between submissions.\n\n"
            )
        elif r != p:  # PIs only: a colleague reviews
            task = self.T(
                f"You are PI {r + 1}. PI {p + 1} has drafted the hypotheses below for the research goal. Review them as a "
                f"colleague at a meeting: for each give specific feedback PI {p + 1} can act on: {crit} Then write one note "
                f"for PI {p + 1}: facts to use, mistakes across the drafts, and overlaps between them.\n\n"
            )
        else:  # one PI, no students: the PI reviews its own drafts
            task = self.T(
                "You are a PI working without students. You drafted the hypotheses below for the research goal. Review them "
                f"critically, as at a meeting: for each give specific feedback you can act on: {crit} Then write one note "
                "for yourself: facts to use, mistakes across the drafts, and overlaps between them.\n\n"
            )
        prompt = (
            task
            + f"RESEARCH GOAL (summary):\n{self.goal_brief(4000)}\n\nSUBMISSIONS:\n{self.submissions(hs)}\n\n"
            + self.data_block(" ".join(f"{h['title']} {h['text'][:1500]}" for h in hs), min(self.data_budget, 10000))
            + 'Return only a JSON object: {"feedback": [{"id": "...", "feedback": "..."}], "lab_note": "..."}'
        )
        d, reply = self.ask_json(f"pi{r + 1} review", prompt, 0.4, dict)
        d = d or {}
        fb = {
            as_text(x.get("id", "")).strip("[] "): as_text(x.get("feedback", ""))
            for x in (d.get("feedback") or [])
            if isinstance(x, dict)
        }
        missing = [h["id"] for h in hs if not fb.get(h["id"])]
        if missing:
            self.errors.append(f"pi{r + 1} review: no feedback for {', '.join(missing)} ({reply_snippet(reply)})")
        for h in hs:
            h["pi_feedback"] = fb.get(h["id"], "")
            if h["pi_feedback"]:
                h["review_comments"].append(h["pi_feedback"])
        note = as_text(d.get("lab_note", ""))
        self.lab_meetings.append(
            dict(pi=p + 1, reviewer=r + 1, n_reviewed=len(hs), n_feedback=len(hs) - len(missing), lab_note=note)
        )
        return note

    def revise(self, args) -> Optional[Dict[str, Any]]:
        h, note = args
        rule = (
            "keep its core idea unless the feedback shows it is wrong, address every point of the feedback, use the facts "
            "in {note}, correct any number that the attached files do not support (say 'not in the files' rather than "
            "invent a number), and keep the full format the goal asks for.\n\n"
        )
        if h["worker"]:
            role = f"student{h['worker']} revision"
            task = self.T(
                f"You are student {h['worker']} in the research group of PI {h['pi']}. Your PI reviewed your hypothesis at "
                "the lab meeting. Revise it: " + rule.format(note="your PI's note to the group")
            )
            labels = self.T("YOUR PI'S FEEDBACK: "), self.T("YOUR PI'S NOTE TO THE GROUP: ")
        else:  # PIs only: the author revises after the review
            role = f"pi{h['pi']} revision"
            r = self.reviewer_of(h["pi"] - 1)
            by = "You reviewed your own draft" if r == h["pi"] - 1 else f"PI {r + 1} reviewed your hypothesis"
            task = self.T(f"You are PI {h['pi']}. {by}. Revise it: " + rule.format(note="the review's note"))
            labels = "FEEDBACK: ", "NOTE: "
        prompt = (
            f"Research Goal: {self.goal}\n\n"
            + task
            + f"HYPOTHESIS: {h['title']}\n{h['text']}\n\n{labels[0]}{h.get('pi_feedback') or '(none)'}\n"
            f"{labels[1]}{note[:2000] or '(none)'}\n\n"
            + self.data_block(f"{h['title']} {h['text'][:4000]}", min(self.data_budget, 12000))
            + "Return only a JSON object with keys 'title' and 'text'."
        )
        d, reply = self.ask_json(role, prompt, 0.6, dict)
        if not d or not d.get("text"):
            self.errors.append(f"{role} of {h['id']}: no parsable rewrite ({reply_snippet(reply)})")
            return None
        return dict(
            h,
            id=self.new_id("V"),
            title=as_text(d.get("title") or h["title"]),
            text=as_text(d["text"]),
            round=2,
            parent_ids=[h["id"]],
            review_comments=list(h["review_comments"]),
            references=list(h["references"]),
            active=True,
        )

    def decide(self, p: int) -> List[str]:
        hs = [h for h in self.hyps.values() if h["active"] and self.group_of(h) == p]
        if not hs:
            return []
        k = max(1, math.ceil(len(hs) * self.forward_share))
        if len(hs) == 1:
            self.decisions.append(
                dict(pi=p + 1, ranking=[hs[0]["id"]], forwarded=[hs[0]["id"]], reason="only submission")
            )
            return [hs[0]["id"]]
        mine = "your group's revised hypotheses the group" if self.n_workers else "your revised hypotheses you"
        prompt = (
            self.T(
                f"You are PI {p + 1}, the principal investigator of a research group. Decide which of {mine} will put "
                "forward. Rank them from strongest to weakest for the research goal, judging fit to the goal, whether the "
                "numbers they quote match the attached files, the strength of the design, and expected impact. The top "
                f"{k} will be put forward"
                + (" to the other PIs, who will compare them with the other groups' ideas" if self.n_pis > 1 else "")
                + ".\n\n"
            )
            + f"RESEARCH GOAL (summary):\n{self.goal_brief(4000)}\n\nHYPOTHESES:\n{self.submissions(hs)}\n\n"
            + self.data_block(" ".join(f"{h['title']} {h['text'][:1500]}" for h in hs), min(self.data_budget, 10000))
            + 'Return only a JSON object: {"ranking": ["id of the strongest", "...", "id of the weakest"], "reason": "..."}'
        )
        d, reply = self.ask_json(f"pi{p + 1} decision", prompt, 0.2, dict)
        d = d or {}
        ids = {h["id"] for h in hs}
        ranking = []
        for x in d.get("ranking") or []:
            x = as_text(x).strip("[] ")
            if x not in ids and x.isdigit() and 1 <= int(x) <= len(hs):  # a position in the list instead of an id
                x = hs[int(x) - 1]["id"]
            if x in ids and x not in ranking:
                ranking.append(x)
        if len(ranking) < len(hs):
            if not ranking:
                self.errors.append(
                    f"pi{p + 1} decision: unusable ranking, submission order kept ({reply_snippet(reply)})"
                )
            ranking += [h["id"] for h in hs if h["id"] not in ranking]
        self.decisions.append(
            dict(pi=p + 1, ranking=ranking, forwarded=ranking[:k], reason=as_text(d.get("reason", "")))
        )
        return ranking

    def judge(self, pair) -> Dict[str, Any]:
        """The swarm's judge prompt, asked of one PI; the PI is recorded with the match."""
        a, b, p = pair
        if self.rng.random() < 0.5:
            a, b = b, a
        prompt = (
            self.T(f"You are PI {p + 1}, sitting on a panel of expert reviewers in a hypothesis tournament. ")
            + "Compare two hypotheses for the research goal.\n"
            "Criteria: fit to the goal; novelty; whether it can be tested with the data the goal describes; whether the data "
            "names, cohort sizes and numbers it quotes are consistent with the goal's information; falsifiability; expected impact.\n"
            "Debate briefly: give the strongest argument for and against each, then decide.\n\n"
            f"RESEARCH GOAL (summary):\n{self.goal_brief(4000)}\n\n"
            f"HYPOTHESIS A: {a['title']}\n{a['text'][:6000]}\n\nHYPOTHESIS B: {b['title']}\n{b['text'][:6000]}\n\n"
            + (
                self.data_block(
                    f"{a['title']} {a['text'][:3000]} {b['title']} {b['text'][:3000]}", min(self.data_budget, 8000)
                )
                + "Check the numbers each hypothesis quotes against the attached data; a number that is not there counts against it.\n\n"
                if self.data
                else ""
            )
            + 'Return only a JSON object: {"for_A": "...", "against_A": "...", "for_B": "...", "against_B": "...", "winner": "A" or "B", "reason": "..."}'
        )
        d = self.ask_json(f"pi{p + 1} judge", prompt, self.judge_t, dict)[0] or {}
        w = str(d.get("winner", "")).strip().upper()[:1]
        if w not in ("A", "B"):
            w = self.rng.choice(["A", "B"])
            d["reason"] = "judge reply not parsable: random winner"
            self.errors.append(f"pi{p + 1} judge: unparsable verdict")
        win, lose = (a, b) if w == "A" else (b, a)
        return dict(
            winner=win["id"],
            loser=lose["id"],
            reason=as_text(d.get("reason", "")),
            against_loser=as_text(d.get(f"against_{'B' if w == 'A' else 'A'}", "")),
            judge_pi=p + 1,
            winner_pi=win["pi"],
            loser_pi=lose["pi"],
            own_group=p + 1 in (win["pi"], lose["pi"]),
            own_group_won=win["pi"] == p + 1 and lose["pi"] != p + 1,
        )

    def pick_judge(self, a: Dict[str, Any], b: Dict[str, Any]) -> int:
        free = [p for p in range(self.n_pis) if p + 1 not in (a["pi"], b["pi"])]
        return self.rng.choice(free or list(range(self.n_pis)))

    def department(self, ids: List[str]) -> None:
        """Swiss rounds over the ideas put forward, as in the swarm's tournament, each match judged by a PI."""
        for m in range(self.matches_per_round):
            act = sorted([self.hyps[i] for i in ids], key=lambda h: -h["elo"])
            if m == 0:
                self.rng.shuffle(act)
            pairs = [(act[i], act[i + 1]) for i in range(0, len(act) - 1, 2)]
            pairs = [(x, y, self.pick_judge(x, y)) for x, y in pairs]
            for res in self.pmap(self.judge, pairs):
                win, lose = self.hyps[res["winner"]], self.hyps[res["loser"]]
                ea = expected(win["elo"], lose["elo"])
                win["elo"] += self.k * (1 - ea)
                lose["elo"] -= self.k * (1 - ea)
                win["wins"] += 1
                lose["losses"] += 1
                self.matches.append(dict(res, round=1, match=m + 1))

    # ------------------------------------------------------------------ the run
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        self.progress(self.T(f"{self.n_pis} PI(s) setting group agendas"))
        for p in range(self.n_pis):
            self.agendas.append(self.agenda(p, list(self.agendas)))
        if self.n_workers:
            jobs = [(s, p, self.agendas[p]["projects"][i]) for p, ss in self.groups.items() for i, s in enumerate(ss)]
        else:  # PIs only
            jobs = [(None, p, self.agendas[p]["projects"][0]) for p in range(self.n_pis)]
        self.progress(self.T(f"{len(jobs)} {'students' if self.n_workers else 'PIs'} proposing in parallel"))
        pool = [h for hs in self.pmap(self.propose, jobs) for h in hs]
        self.progress(f"proximity: de-duplicating {len(pool)} hypotheses")
        self.dedupe(pool)
        forwarded: List[str] = []
        rest: List[str] = []
        if not pool:
            self.errors.append("no hypotheses were generated: every student reply was empty or unreadable")
            self.progress("stopped: no hypotheses to review")
        else:
            self.progress(self.T("lab meetings: each PI reviews the group's submissions"))
            notes = self.pmap(self.lab_meeting, list(range(self.n_pis)))
            todo = [h for h in self.hyps.values() if h["active"]]
            self.progress(self.T(f"{'students' if self.n_workers else 'PIs'} revising {len(todo)} hypotheses"))

            def heard(h):  # the lab note of the idea's PI, and of the student's other advisors when co-advised
                if not h["worker"]:
                    return notes[self.group_of(h)]
                adv = self.advisors(h["worker"] - 1)
                if len(adv) == 1:
                    return notes[adv[0]]
                own = self.group_of(h)
                return "\n\n".join(
                    self.T(f"PI {q + 1}: ") + notes[q] for q in [own] + [q for q in adv if q != own] if notes[q]
                )

            for old, new in zip(todo, self.pmap(self.revise, [(h, heard(h)) for h in todo])):
                if new:
                    old["active"], old["revised_into"] = False, new["id"]
                    self.hyps[new["id"]] = new
            self.progress(self.T("PIs deciding what each group puts forward"))
            rankings = self.pmap(self.decide, list(range(self.n_pis)))
            for p, ranking in enumerate(rankings):
                k = max(1, math.ceil(len(ranking) * self.forward_share)) if ranking else 0
                for pos, i in enumerate(ranking, 1):
                    self.hyps[i]["pi_rank"], self.hyps[i]["forwarded"] = pos, pos <= k
                forwarded += ranking[:k]
            rest = sorted(
                (i for r in rankings for i in r if i not in forwarded),
                key=lambda i: (self.hyps[i]["pi_rank"], self.hyps[i]["pi"]),
            )
            if len(forwarded) > 1:
                self.progress(
                    self.T(f"PIs comparing {len(forwarded)} ideas put forward ({self.matches_per_round} Swiss rounds)")
                )
                self.department(forwarded)
        ranked = sorted(forwarded, key=lambda i: -self.hyps[i]["elo"])
        floor = min((self.hyps[i]["elo"] for i in ranked), default=1200.0)
        for n, i in enumerate(rest, 1):  # ideas not put forward rank below every forwarded idea, in their PI's order
            self.hyps[i]["elo"] = floor - 10.0 * n
        final = (ranked + rest)[: self.final_k]
        if self.data:
            for h in self.hyps.values():
                h["data_check"] = self.data.check_numbers(h["title"] + "\n" + h["text"])
        self.progress(f"done: {len(self.trace)} LLM calls in {time.time() - t0:.0f}s")
        return dict(
            settings=dict(
                model=self.model,
                mode="research_group",
                framing=self.framing,
                n_pis=self.n_pis,
                n_students=self.n_workers,
                n_workers=self.n_workers,
                per_worker=self.per_worker,
                pi_model=self.pi_model,
                student_model=self.student_model,
                forward_share=self.forward_share,
                rounds=1,
                matches_per_round=self.matches_per_round,
                final_k=self.final_k,
                dedupe_threshold=self.dedupe_threshold,
                parallel=self.parallel,
                k_factor=self.k,
                data=self.data.summary() if self.data else None,
                web=None,
            ),
            groups={str(p + 1): [s + 1 for s in ss] for p, ss in self.groups.items()},
            agendas=self.agendas,
            lenses=[x for a in self.agendas for x in a["projects"]],
            lab_meetings=self.lab_meetings,
            decisions=self.decisions,
            web_search=[],
            hypotheses=list(self.hyps.values()),
            final=final,
            merged=self.merged,
            matches=self.matches,
            meta_reviews=[],
            errors=self.errors,
            seconds=round(time.time() - t0, 1),
            trace=self.trace,
        )
