"""Team lab: an MCP server that simulates how teams of specialists and teams of generalists complete multi-skill tasks.

It answers goals such as "are teams of experts with one skill each more effective than teams of generalists who need
50% more time per task?" by Monte Carlo simulation instead of by argument. Each task needs a number of skill areas
(1 + Poisson(mean_skills - 1), capped at the number of areas); the time for each skill area is drawn from a lognormal
distribution around hours_per_skill. A specialist team has one member per skill area: a task passes from specialist to
specialist, with a handoff delay between consecutive areas, and each specialist works on one task at a time. A
generalist team has the same number of members; any member can do every area of a task but takes
generalist_time_factor times as long, and needs no handoffs. Tasks arrive together and are assigned first come, first
served, so the simulation reports both the time to finish one task and the time for the team to finish all of them.

    python -m app.sim_servers.team_lab        # stdio MCP server (the app starts it)
"""

from __future__ import annotations

import heapq
import math
import random
from typing import Any, Dict, List

from mcp.server.mcpserver import MCPServer

server = MCPServer("team-lab")


def _skills(rng: random.Random, mean_skills: float, areas: int) -> List[int]:
    lam, k, u = max(mean_skills - 1, 0), 0, rng.random()
    term = math.exp(-lam)  # Poisson by inversion
    cdf = term
    while u > cdf and k < 50:
        k += 1
        term *= lam / k
        cdf += term
    n = min(1 + k, areas)
    return rng.sample(range(areas), n)


def _simulate(n_tasks, mean_skills, hours_per_skill, generalist_time_factor, handoff_hours, team_size, spread, seed):
    rng = random.Random(seed)
    sigma = math.sqrt(math.log(1 + spread**2))
    mu = math.log(hours_per_skill) - sigma**2 / 2
    tasks = []
    for _ in range(n_tasks):
        areas = _skills(rng, mean_skills, team_size)
        tasks.append([(a, rng.lognormvariate(mu, sigma)) for a in areas])
    # specialists: each area is one person; a task visits its areas in order, and each specialist takes the task that
    # became ready first (event-driven, so no specialist idles while a ready task waits)
    free = [0.0] * team_size
    spec_done = [0.0] * len(tasks)
    spec_task = [sum(h for _, h in t) + handoff_hours * (len(t) - 1) for t in tasks]
    ready = [(0.0, i, 0) for i in range(len(tasks))]  # (ready time, task, step)
    heapq.heapify(ready)
    while ready:
        t_ready, i, step = heapq.heappop(ready)
        a, h = tasks[i][step]
        finish = max(t_ready, free[a]) + h
        free[a] = finish
        if step + 1 < len(tasks[i]):
            heapq.heappush(ready, (finish + handoff_hours, i, step + 1))
        else:
            spec_done[i] = finish
    # generalists: the next free member does the whole task, generalist_time_factor times slower
    heap = [0.0] * team_size
    gen_done, gen_task = [], []
    for t in tasks:
        start = heapq.heappop(heap)
        dur = generalist_time_factor * sum(h for _, h in t)
        heapq.heappush(heap, start + dur)
        gen_done.append(start + dur)
        gen_task.append(dur)
    return dict(
        spec_task=spec_task,
        gen_task=gen_task,
        spec_makespan=max(spec_done),
        gen_makespan=max(gen_done),
        skills=[len(t) for t in tasks],
    )


@server.tool()
def simulate_teams(
    n_tasks: int = 500,
    mean_skills: float = 3.0,
    hours_per_skill: float = 8.0,
    generalist_time_factor: float = 1.5,
    handoff_hours: float = 2.0,
    team_size: int = 4,
    spread: float = 0.5,
    replicates: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """Monte Carlo comparison of a specialist team and a generalist team of the same size on the same tasks.
    Returns the mean hours to finish one task, the hours for the team to finish all tasks (throughput), and the
    percentage by which the specialist team is faster on each, with a 95% interval across replicates."""
    n_tasks, team_size, replicates = (
        max(1, min(int(n_tasks), 5000)),
        max(1, min(int(team_size), 20)),
        max(1, min(int(replicates), 200)),
    )
    per_task, makespan, skills = [], [], []
    for r in range(replicates):
        s = _simulate(
            n_tasks, mean_skills, hours_per_skill, generalist_time_factor, handoff_hours, team_size, spread, seed + r
        )
        st, gt = sum(s["spec_task"]) / n_tasks, sum(s["gen_task"]) / n_tasks
        per_task.append((st, gt, (gt - st) / gt * 100))
        makespan.append(
            (s["spec_makespan"], s["gen_makespan"], (s["gen_makespan"] - s["spec_makespan"]) / s["gen_makespan"] * 100)
        )
        skills.append(sum(s["skills"]) / n_tasks)

    def summary(rows):
        d = sorted(x[2] for x in rows)
        lo, hi = d[int(0.025 * (len(d) - 1))], d[int(round(0.975 * (len(d) - 1)))]
        return dict(
            specialist_hours=round(sum(x[0] for x in rows) / len(rows), 2),
            generalist_hours=round(sum(x[1] for x in rows) / len(rows), 2),
            specialists_faster_pct=round(sum(d) / len(d), 1),
            interval_95=[round(lo, 1), round(hi, 1)],
        )

    breakeven = (generalist_time_factor - 1) * hours_per_skill * mean_skills / max(mean_skills - 1, 1e-9)
    return dict(
        settings=dict(
            n_tasks=n_tasks,
            mean_skills=mean_skills,
            hours_per_skill=hours_per_skill,
            generalist_time_factor=generalist_time_factor,
            handoff_hours=handoff_hours,
            team_size=team_size,
            spread=spread,
            replicates=replicates,
        ),
        mean_skill_areas_per_task=round(sum(skills) / len(skills), 2),
        one_task=summary(per_task),
        all_tasks=summary(makespan),
        handoff_hours_at_which_one_task_takes_equally_long=round(breakeven, 2),
    )


if __name__ == "__main__":
    server.run()
