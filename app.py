import logging
import os
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

from app.agents import SupervisorAgent

# Import the existing app components
from app.config import config
from app.data_context import DataBundle, data_check_html
from app.export import directions_from_lenses, ideas_from_hypotheses, write_bundle
from app.models import ContextMemory, ResearchGoal
from app.peer_review import PeerReviewCoScientist
from app.research_group import ResearchGroupCoScientist
from app.run_store import delete_run, get_reports_dir, history_html, list_runs, report_file_url, save_run, write_report
from app.science_tools import ScienceTools, tool_choices
from app.simulation import SimulationLab, format_simulation_html, server_choices, server_from_choice
from app.swarm import SwarmCoScientist, format_swarm_results
from app.tools.arxiv_search import ArxivSearchTool
from app.utils import (
    FALLBACK_MIN_PARAMS_B,
    _estimate_model_size_b,
    classify_llm_error,
    fetch_free_models,
    get_deployment_environment,
    is_huggingface_space,
    logger,
    order_free_models_for_demo,
)
from app.web_search import WebSearch

# Global state for the Gradio app
global_context = ContextMemory()
supervisor = SupervisorAgent()
current_research_goal: Optional[ResearchGoal] = None
available_models: List[str] = []
CONFIGURED_LLM_MODEL = config.get("llm_model", "")
SAFE_FALLBACK_LLM_MODEL = CONFIGURED_LLM_MODEL or "-- Select Model --"
CYCLE_TIMEOUT_SECONDS = int(
    os.getenv("CO_SCIENTIST_CYCLE_TIMEOUT_SECONDS", "1200")
)  # same as swarm: busy free models need retries
CYCLE_PROGRESS_INTERVAL_SECONDS = 5
SWARM_TIMEOUT_SECONDS = int(os.getenv("CO_SCIENTIST_SWARM_TIMEOUT_SECONDS", "1200"))
TEAM_TIMEOUT_SECONDS = int(os.getenv("CO_SCIENTIST_TEAM_TIMEOUT_SECONDS", "2400"))  # team modes make more, longer calls
LAST_SWARM: Dict[str, Any] = {}
# Organized swarms (app/research_group.py, app/peer_review.py): the same workers given roles
TEAM_MODES = ("Research group", "Startup team", "Peer review")
SAME_MODEL = "Same as LLM Model"

# Configure logging for Gradio
logging.basicConfig(level=logging.INFO)


def fetch_available_models():
    """Fetch available models from OpenRouter with environment-based filtering."""
    global available_models

    # Detect deployment environment
    deployment_env = get_deployment_environment()
    is_hf_spaces = is_huggingface_space()

    logger.info(f"Detected deployment environment: {deployment_env}")
    logger.info(f"Is Hugging Face Spaces: {is_hf_spaces}")

    try:
        # Apply filtering based on environment
        if is_hf_spaces:
            # Use only dynamically checked free models for Hugging Face Spaces.
            available_models = fetch_free_models() or ([CONFIGURED_LLM_MODEL] if CONFIGURED_LLM_MODEL else [])
            logger.info(f"Hugging Face Spaces: Filtered to {len(available_models)} free models")
        else:
            response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
            response.raise_for_status()
            models_data = response.json().get("data", [])

            # Extract all model IDs
            all_models = sorted([model.get("id") for model in models_data if model.get("id")])

            # Use all models in local/development environment
            available_models = all_models
            logger.info(f"Local/Development: Using all {len(available_models)} models")

    except Exception as e:
        logger.error(f"Failed to fetch models from OpenRouter: {e}")
        cached_free_models = fetch_free_models()
        available_models = cached_free_models or ([SAFE_FALLBACK_LLM_MODEL] if SAFE_FALLBACK_LLM_MODEL else [])

    return available_models


def get_default_model_choice(models: Optional[List[str]] = None) -> str:
    """Prefer the configured free model when live, then a fast free model."""
    model_choices = models or available_models
    if (
        CONFIGURED_LLM_MODEL
        and ":free" in CONFIGURED_LLM_MODEL
        and (not model_choices or CONFIGURED_LLM_MODEL in model_choices)
    ):
        return CONFIGURED_LLM_MODEL
    free_models = order_free_models_for_demo(model_choices)
    capable = [m for m in free_models if (_estimate_model_size_b(m) or FALLBACK_MIN_PARAMS_B) >= FALLBACK_MIN_PARAMS_B]
    if capable or free_models:
        return (capable or free_models)[0]
    return CONFIGURED_LLM_MODEL or SAFE_FALLBACK_LLM_MODEL


def get_model_dropdown_choices(models: Optional[List[str]] = None) -> List[str]:
    """Return model choices with a cost-safe default first and de-duplicated."""
    model_choices = models or available_models
    choices = [get_default_model_choice(model_choices)]
    ordered = order_free_models_for_demo(model_choices)
    small = [m for m in ordered if (_estimate_model_size_b(m) or FALLBACK_MIN_PARAMS_B) < FALLBACK_MIN_PARAMS_B]
    # capable free models first, very small ones (under 12B) last: they rarely follow a long goal
    for model in [*[m for m in ordered if m not in small], *small, *model_choices]:
        if model and model not in choices:
            choices.append(model)
    return choices


def get_deployment_status():
    """Get deployment status information."""
    deployment_env = get_deployment_environment()
    is_hf_spaces = is_huggingface_space()

    if is_hf_spaces:
        status = (
            f"🚀 Running in {deployment_env} | Models filtered for cost control ({len(available_models)} available)"
        )
        color = "orange"
    else:
        status = f"💻 Running in {deployment_env} | All models available ({len(available_models)} total)"
        color = "blue"

    return status, color


def history_run_choices() -> List[Tuple[str, str]]:
    """Return dropdown choices for deleting saved runs."""
    choices = []
    for run in list_runs(limit=None):
        goal = run.get("goal") or "Untitled goal"
        if len(goal) > 80:
            goal = f"{goal[:77]}..."
        label = f"{run.get('created_at') or 'Unknown date'} — {goal} ({run.get('run_id')})"
        choices.append((label, run.get("run_id")))
    return choices


def refresh_history_view() -> Tuple[str, Dict[str, Any], str]:
    """Refresh the history table and delete-run dropdown."""
    return history_html(), gr.update(choices=history_run_choices(), value=None), ""


def delete_history_run(selected_run_id: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    """Delete the selected saved run and refresh the history display."""
    if not selected_run_id:
        return "Select a saved run to delete.", history_html(), gr.update(choices=history_run_choices(), value=None)

    deleted = delete_run(selected_run_id)
    message = f"Deleted saved run {selected_run_id}." if deleted else f"Saved run {selected_run_id} was not found."
    return message, history_html(), gr.update(choices=history_run_choices(), value=None)


TOOL_MODES = ["No tools", "All tools", "Choose tools"]
WEB_SEARCH_TOOL = "web_search"


def build_tools(mode: str, picks: Optional[List[str]] = None) -> Tuple[Optional[ScienceTools], Optional[WebSearch]]:
    """The run's science tools and web search from the UI choice; (None, None) for No tools, so prompts are unchanged."""
    if mode == "All tools":
        return ScienceTools(), WebSearch()
    if mode == "Choose tools":
        picks = list(picks or [])
        names = [p for p in picks if p != WEB_SEARCH_TOOL]
        return (ScienceTools(names) if names else None), (WebSearch() if WEB_SEARCH_TOOL in picks else None)
    return None, None


def build_simulation(choice: str, data_paths: Optional[List[str]] = None) -> Optional[SimulationLab]:
    """The run's simulation check from the UI choice; None for No simulation, so runs are unchanged. The records lab
    reads the attached tables, copied into a folder of their own."""
    key = server_from_choice(choice or "")
    if not key:
        return None
    data_dir = None
    if key == "records":
        import shutil
        import tempfile

        tables = [p for p in data_paths or [] if str(p).lower().endswith((".csv", ".tsv"))]
        if not tables:
            raise ValueError("The records lab computes from attached tables: attach .csv or .tsv files first.")
        data_dir = tempfile.mkdtemp(prefix="records_lab_")
        for p in tables:
            shutil.copy(p, os.path.join(data_dir, os.path.basename(p)))
    return SimulationLab(key, data_dir=data_dir)


def simulation_check_single(sim: SimulationLab, goal: ResearchGoal, context: ContextMemory) -> str:
    """Check the top single-mode hypotheses with the simulation server; return the HTML section."""
    from app.utils import call_llm

    hs = sorted(context.hypotheses.values(), key=lambda h: -(h.elo_score or 0))
    ideas = [dict(id=h.hypothesis_id, title=h.title, text=h.text, elo=h.elo_score) for h in hs]
    data = getattr(goal, "data", None)
    res = sim.check(
        lambda p: call_llm(p, temperature=0.3, model=goal.llm_model),
        goal.description,
        ideas,
        data=data.outline() if data else "",
    )
    return format_simulation_html(res) or "<p>Simulation check: no idea could be checked.</p>"


def tools_summary(tools: Optional[ScienceTools], web: Optional[WebSearch]) -> str:
    parts = ([f"{len(tools.enabled)} science tools"] if tools else []) + (["web search"] if web else [])
    return " + ".join(parts) if parts else "none"


def set_research_goal(
    description: str,
    llm_model: str = None,
    num_hypotheses: int = 3,
    generation_temperature: float = 0.7,
    reflection_temperature: float = 0.5,
    elo_k_factor: int = 32,
    top_k_hypotheses: int = 2,
    data=None,
    tools=None,
    web=None,
) -> Tuple[str, str]:
    """Set the research goal and initialize the system."""
    global current_research_goal, global_context

    if not description.strip():
        return "❌ Error: Please enter a research goal.", ""

    try:
        # Create research goal with settings
        current_research_goal = ResearchGoal(
            description=description.strip(),
            constraints={},
            llm_model=llm_model if llm_model and llm_model != "-- Select Model --" else None,
            num_hypotheses=num_hypotheses,
            generation_temperature=generation_temperature,
            reflection_temperature=reflection_temperature,
            elo_k_factor=elo_k_factor,
            top_k_hypotheses=top_k_hypotheses,
            data=data,
            tools=tools,
            web=web,
        )

        # Reset context
        global_context = ContextMemory()

        logger.info(f"Research goal set: {description}")
        logger.info(f"Settings: model={current_research_goal.llm_model}, num={current_research_goal.num_hypotheses}")

        status_msg = f"✅ Research goal set successfully!\n\n**Goal:** {description}\n**Model:** {current_research_goal.llm_model or 'Default'}\n**Hypotheses per cycle:** {num_hypotheses}"
        if data:
            status_msg += f"\n**Data:** {data.summary()}"
        if tools or web:
            status_msg += f"\n**Tools:** {tools_summary(tools, web)}"

        return status_msg, "Ready to run first cycle. Click 'Run Cycle' to begin."

    except Exception as e:
        error_msg = f"❌ Error setting research goal: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""


def execute_cycle(
    research_goal: ResearchGoal,
    context: ContextMemory,
    cycle_supervisor: SupervisorAgent,
) -> Dict[str, Any]:
    """Run a cycle against the supplied state and return display-ready results."""
    import datetime

    # Prepare log file
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"app_log_{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write(f"LOGGING FOR THIS GOAL: {research_goal.description}\n")
        f.write("--- Endpoint /run_cycle START ---\n")

    try:
        iteration = context.iteration_number + 1
        logger.info(f"Running cycle {iteration}")

        # Run the cycle
        cycle_details = cycle_supervisor.run_cycle(research_goal, context)

        # Log all steps and hypotheses
        steps = cycle_details.get("steps", {})
        with open(log_file, "a") as f:
            for step_name, step_data in steps.items():
                hypos = step_data.get("hypotheses", [])
                f.write(f"Step: {step_name} | {len(hypos)} hypotheses\n")
                for h in hypos:
                    f.write(f"  - ID: {h.get('id')} | Title: {h.get('title')} | Elo: {h.get('elo_score', 'N/A')}\n")

        # Uploaded data: check every quoted number of every hypothesis against the files
        data = getattr(research_goal, "data", None)
        if data:
            for step in cycle_details.get("steps", {}).values():
                for h in step.get("hypotheses", []) or []:
                    if isinstance(h, dict):
                        h["data_check"] = data.check_numbers(f"{h.get('title', '')}\n{h.get('text', '')}")
            cycle_details["data"] = data.summary()

        # Format results for display (also logs final rankings)
        results_html = format_cycle_results(cycle_details, log_file=log_file)

        # Get references
        references_html = get_references_html(cycle_details, research_goal=research_goal)

        # Status message: surface the real cause when generation failed, instead
        # of reporting success over an empty result (issue llnl#36).
        errors = cycle_details.get("errors", [])
        produced_any = bool(cycle_details.get("steps", {}).get("generation", {}).get("hypotheses"))
        if errors:
            categories = sorted({classify_llm_error(e) for e in errors})
            cause = "; ".join(categories)
            if produced_any:
                status_msg = f"⚠️ Cycle {iteration} completed with errors ({cause}). Log: {log_file}"
            else:
                status_msg = (
                    f"⚠️ Cycle {iteration} could not generate hypotheses — {cause}. "
                    f"See the results panel for details. Log: {log_file}"
                )
        else:
            status_msg = f"✅ Cycle {iteration} completed successfully! Log: {log_file}"

        return {
            "status": status_msg,
            "results_html": results_html,
            "references_html": references_html,
            "cycle_details": cycle_details,
            "log_file": log_file,
        }

    except Exception as e:
        error_msg = f"❌ Error during cycle execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": error_msg,
            "results_html": "",
            "references_html": "",
            "cycle_details": {"iteration": context.iteration_number + 1, "steps": {}, "errors": [error_msg]},
            "log_file": log_file,
        }


def persist_cycle_result(research_goal: ResearchGoal, cycle_result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Persist an accepted cycle result and return Gradio output values."""
    tools = getattr(research_goal, "tools", None)
    if tools:  # every tool call this run made, kept with the run
        cycle_result["cycle_details"]["tool_calls"] = tools.log
    saved_run = save_run(
        research_goal=research_goal,
        cycle_details=cycle_result["cycle_details"],
        status=cycle_result["status"],
        references_html=cycle_result["references_html"],
        results_html=cycle_result["results_html"],
        log_file=cycle_result["log_file"],
    )
    report_path = write_report(saved_run)
    status_msg = f"{cycle_result['status']}\nRun ID: {saved_run['run_id']}\nReport: {report_file_url(report_path)}"
    return status_msg, cycle_result["results_html"], cycle_result["references_html"]


def run_cycle() -> Tuple[str, str, str]:
    """Run a single research cycle with detailed step logging for debugging."""
    global current_research_goal, global_context, supervisor

    if not current_research_goal:
        return "❌ Error: No research goal set. Please set a research goal first.", "", ""

    return persist_cycle_result(
        current_research_goal,
        execute_cycle(current_research_goal, global_context, supervisor),
    )


def format_timeout_duration(timeout_seconds: float) -> str:
    if timeout_seconds < 60:
        return f"{timeout_seconds:g} seconds"
    minutes = timeout_seconds / 60
    if minutes.is_integer():
        return f"{int(minutes)} minutes"
    return f"{minutes:.1f} minutes"


def timeout_results_html(timeout_seconds: float) -> str:
    timeout_duration = format_timeout_duration(timeout_seconds)
    return f"""
    <div style="margin: 20px 0; padding: 15px; border: 2px solid #e67e22; border-radius: 8px; background-color: rgba(230, 126, 34, 0.12); color: inherit;">
        <h3>Cycle stopped at the time limit</h3>
        <p>The run exceeded the {timeout_duration} upper limit before the app received a completed cycle.</p>
        <p>Try fewer hypotheses, a different model, or a later retry if the model provider is slow.</p>
    </div>
    """


def run_cycle_with_progress(
    timeout_seconds: int = CYCLE_TIMEOUT_SECONDS,
    poll_seconds: float = CYCLE_PROGRESS_INTERVAL_SECONDS,
):
    """Run a cycle in the background and stream status updates until done or timed out."""
    global global_context

    if not current_research_goal:
        yield "❌ Error: No research goal set. Please set a research goal first.", "", ""
        return

    run_goal = current_research_goal
    run_context = deepcopy(global_context)
    run_supervisor = SupervisorAgent()
    result: Dict[str, Dict[str, Any]] = {}

    def worker():
        result["value"] = execute_cycle(run_goal, run_context, run_supervisor)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    iteration = global_context.iteration_number + 1

    while thread.is_alive():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            timeout_duration = format_timeout_duration(timeout_seconds)
            timeout_status = (
                f"⚠️ Cycle {iteration} timed out after {timeout_duration}. "
                "The app stopped waiting for the model provider instead of leaving the run spinning."
            )
            timeout_html = timeout_results_html(timeout_seconds)
            saved_run = save_run(
                research_goal=run_goal,
                cycle_details={
                    "iteration": iteration,
                    "steps": {},
                    "errors": [timeout_status],
                },
                status=timeout_status,
                references_html="",
                results_html=timeout_html,
                log_file="",
            )
            report_path = write_report(saved_run)
            yield (
                f"{timeout_status}\nRun ID: {saved_run['run_id']}\nReport: {report_file_url(report_path)}",
                timeout_html,
                "",
            )
            return

        status = (
            f"⏳ Cycle {iteration} is running.\n"
            f"Elapsed: {format_timeout_duration(elapsed)}.\n"
            "Active work: generating, reviewing, ranking, and evolving hypotheses.\n"
            f"Upper limit: {format_timeout_duration(timeout_seconds)}."
        )
        yield status, "<p>Cycle is still running. Results will appear when the current step completes.</p>", ""
        thread.join(timeout=min(poll_seconds, max(timeout_seconds - elapsed, 0.1)))

    cycle_result = result.get("value")
    if not cycle_result:
        yield "❌ Error: Cycle ended without a result.", "", ""
        return
    if current_research_goal is run_goal:
        global_context = run_context
    yield persist_cycle_result(run_goal, cycle_result)


def run_swarm_with_progress(
    goal: str,
    llm_model: Optional[str],
    n_workers: int,
    per_worker: int,
    rounds: int,
    matches: int,
    timeout_seconds: int = SWARM_TIMEOUT_SECONDS,
    poll_seconds: float = CYCLE_PROGRESS_INTERVAL_SECONDS,
    data=None,
    tools=None,
    web=None,
    sim=None,
):
    """Run swarm mode in the background and stream its progress until done or timed out."""
    model = llm_model if llm_model and llm_model != "-- Select Model --" else None
    log: List[str] = []
    swarm = SwarmCoScientist(
        goal,
        model=model,
        n_workers=int(n_workers),
        per_worker=int(per_worker),
        rounds=int(rounds),
        matches_per_round=int(matches),
        top_k_evolve=max(1, min(3, int(n_workers) * int(per_worker) // 2)),
        final_k=min(10, int(n_workers) * int(per_worker)),
        parallel=min(4, int(n_workers)),
        progress=log.append,
        data=data,
        tools=tools,
        web=web,
        sim=sim,
    )
    yield from stream_swarm_run(swarm, goal, model, log, timeout_seconds, poll_seconds)


def stream_swarm_run(
    swarm,
    goal: str,
    model: Optional[str],
    log: List[str],
    timeout_seconds: int = SWARM_TIMEOUT_SECONDS,
    poll_seconds: float = CYCLE_PROGRESS_INTERVAL_SECONDS,
    label: str = "Swarm",
    organization: str = "",
):
    """Run a swarm (flat or organized) in the background; stream progress, then save and show the result."""
    result: Dict[str, Any] = {}

    def worker():
        try:
            result["value"] = swarm.run()
        except Exception as e:  # surface the cause instead of a silent empty panel
            logger.error("%s run failed: %s", label, e, exc_info=True)
            result["error"] = str(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    while thread.is_alive():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            yield (
                f"⚠️ {label} stopped waiting after {format_timeout_duration(timeout_seconds)}.",
                timeout_results_html(timeout_seconds),
                "",
            )
            return
        steps = "\n".join(log[-4:])
        yield (
            f"⏳ {label} running ({format_timeout_duration(elapsed)}, {len(swarm.trace)} LLM calls so far)\n{steps}",
            f"<p>{label} is running. Results appear when the final ranking is done.</p>",
            "",
        )
        thread.join(timeout=min(poll_seconds, max(timeout_seconds - elapsed, 0.1)))
    if "error" in result or "value" not in result:
        yield f"❌ {label} failed: {result.get('error', 'no result')}", "", ""
        return
    res = result["value"]
    LAST_SWARM.clear()
    LAST_SWARM.update(res=res, goal=goal, model=model, mode=label.lower())
    html_out = format_swarm_results(res)
    if organization:
        html_out = f"<p><b>{label}:</b> {organization}</p>" + html_out
    if label == "Swarm":
        status = f"✅ Swarm finished: {len(res['final'])} hypotheses ranked from {len(res['hypotheses'])} after {len(res['matches'])} debates."
    else:
        status = f"✅ {label} finished: {len(res['final'])} hypotheses ranked from {len(res['hypotheses'])} ({organization})."
    if res["errors"]:
        status += f" {len(res['errors'])} problems (see results)."
    saved = save_run(
        research_goal=ResearchGoal(description=goal, llm_model=model),
        cycle_details={
            "iteration": 1,
            "mode": label.lower(),
            "steps": {},
            "swarm": {k: v for k, v in res.items() if k != "trace"},
        },
        status=status,
        references_html="",
        results_html=html_out,
        log_file="",
    )
    report_path = write_report(saved)
    yield f"{status}\nRun ID: {saved['run_id']}\nReport: {report_file_url(report_path)}", html_out, ""


def per_member(total: int, members: int) -> int:
    """Ideas per member so that the team proposes about `total` ideas in all."""
    return max(1, int(total) // max(1, int(members)))


def role_model(choice: Optional[str], model: Optional[str]) -> Optional[str]:
    return model if not choice or choice in (SAME_MODEL, "-- Select Model --") else choice


def build_team(
    mode: str,
    goal: str,
    model: Optional[str],
    lab_pis: int = 1,
    lab_students: int = 4,
    lab_total: int = 8,
    lab_pi_model: Optional[str] = None,
    lab_student_model: Optional[str] = None,
    biz_founders: int = 2,
    biz_juniors: int = 0,
    biz_total: int = 8,
    rev_authors: int = 4,
    rev_reviewers: int = 3,
    rev_total: int = 8,
    rev_reviewer_model: Optional[str] = None,
    data=None,
    progress=None,
):
    """The organized swarm for a team mode and a one-line description of it; ValueError for an invalid team."""
    kw = dict(data=data, progress=progress or (lambda m: None), parallel=4)
    if mode == "Peer review":
        a, r = int(rev_authors), int(rev_reviewers)
        team = PeerReviewCoScientist(
            goal,
            model=model,
            n_authors=a,
            n_reviewers=r,
            per_author=per_member(rev_total, a),
            reviewer_model=role_model(rev_reviewer_model, model),
            **kw,
        )
        return team, f"{a} author{'s' * (a > 1)}, {r} reviewer{'s' * (r > 1)}, two review rounds"
    if mode in ("Research group", "Startup team"):
        lab = mode == "Research group"
        p, s = (int(lab_pis), int(lab_students)) if lab else (int(biz_founders), int(biz_juniors))
        if lab and s < 1:
            raise ValueError("A research group needs at least one student.")
        if s and s % p and p % s:
            who = "students and PIs" if lab else "junior members and co-founders"
            raise ValueError(f"The numbers of {who} must divide one another (for example 2 and 4, or 4 and 1).")
        team = ResearchGroupCoScientist(
            goal,
            model=model,
            n_pis=p,
            n_students=s,
            per_student=per_member(lab_total if lab else biz_total, s or p),
            pi_model=role_model(lab_pi_model, model) if lab else model,
            student_model=role_model(lab_student_model, model) if lab else model,
            framing="lab" if lab else "startup",
            **kw,
        )
        if lab:
            return team, f"{p} PI{'s' * (p > 1)}, {s} student{'s' * (s > 1)}"
        return team, f"{p} co-founder{'s' * (p > 1)}" + (
            f", {s} junior member{'s' * (s > 1)}" if s else " (no junior members)"
        )
    raise ValueError(f"Unknown team mode: {mode}")


def mode_panels(mode: str):
    """Show only the settings panel of the chosen mode."""
    return [gr.update(visible=mode == m) for m in ("Swarm", *TEAM_MODES)]


def export_bundle_zip(mode: str) -> Optional[str]:
    """Export results of the latest run (.md, .html, .json, reviews, references); return the zip path."""
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = get_reports_dir() / "bundles" / f"coscientist_{'swarm' if mode == 'Swarm' else 'single'}_{stamp}"
    try:
        if mode == "Swarm":
            if not LAST_SWARM:
                return None
            res = LAST_SWARM["res"]
            by = {h["id"]: h for h in res["hypotheses"]}
            final = [by[i] for i in res["final"]]
            ideas = ideas_from_hypotheses(final, res["matches"])
            note = (
                f"research goal + uploaded data: {res['settings']['data']}"
                if res["settings"].get("data")
                else "the research-goal text only (no files)"
            )
            return write_bundle(
                ideas,
                str(out),
                out.name,
                LAST_SWARM["goal"],
                directions=directions_from_lenses(res.get("lenses")),  # the records the live Directions list shows
                tool_calls=res.get("tool_calls"),
                mode=LAST_SWARM.get("mode") or "swarm",
                model=LAST_SWARM.get("model") or "",
                input_note=note,
            )
        if not current_research_goal or not global_context.hypotheses:
            return None
        hyps = [h.to_dict() for h in global_context.hypotheses.values()]
        data = getattr(current_research_goal, "data", None)
        if data:
            for h in hyps:
                h["data_check"] = data.check_numbers(f"{h.get('title', '')}\n{h.get('text', '')}")
        ideas = ideas_from_hypotheses(hyps)
        return write_bundle(
            ideas,
            str(out),
            out.name,
            current_research_goal.description,
            tool_calls=current_research_goal.tools.log if getattr(current_research_goal, "tools", None) else None,
            mode="single",
            model=current_research_goal.llm_model or "",
            input_note=f"research goal + uploaded data: {data.summary()}"
            if data
            else "the research-goal text only (no files)",
        )
    except Exception as e:
        logger.error("Bundle export failed: %s", e, exc_info=True)
        return None


def format_cycle_results(cycle_details: Dict, log_file: str = None) -> str:
    """Format cycle results as HTML with expandable sections. Optionally log final rankings to log_file."""
    import html as html_lib

    html = f"<h2>🔬 Iteration {cycle_details.get('iteration', 'Unknown')}</h2>"

    # Surface generation errors up front with an actionable category, so a failed
    # run explains itself instead of silently showing empty rankings (issue llnl#36).
    errors = cycle_details.get("errors", [])
    if errors:
        items = ""
        for e in errors:
            category = classify_llm_error(e)
            items += f"<li><strong>{html_lib.escape(category)}:</strong> {html_lib.escape(str(e))}</li>"
        html += f"""
        <div style="margin: 20px 0; padding: 15px; border: 2px solid #e74c3c; border-radius: 8px; background-color: rgba(231, 76, 60, 0.12); color: inherit;">
            <h3>⚠️ Generation could not complete</h3>
            <p>The model/API reported the following, so some or all hypotheses were not generated:</p>
            <ul style="color: #c0392b;">{items}</ul>
        </div>
        """

    # Process steps in order
    steps = cycle_details.get("steps", {})
    # Display steps in the order they appear in the steps dict (preserves backend execution order)
    for step_name, step_data in steps.items():
        step_title = {
            "generation": "🎯 Generation",
            "reflection": "🔍 Reflection",
            "ranking": "📊 Ranking",
            "evolution": "🧬 Evolution",
            "reflection_evolved": "🔍 Reflection (Evolved)",
            "ranking_final": "📊 Final Ranking",
            "proximity": "🔗 Proximity Analysis",
            "meta_review": "📋 Meta-Review",
        }.get(step_name, step_name.title())

        html += f"""
        <details style="margin: 15px 0; border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
            <summary style="font-weight: bold; font-size: 1.1em; cursor: pointer; padding: 5px;">
                {step_title}
            </summary>
            <div style="margin-top: 10px; padding: 10px; background-color: rgba(127, 127, 127, 0.10); color: inherit; border-radius: 5px;">
        """

        # Step-specific content
        if step_name == "generation":
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Generated {len(hypotheses)} new hypotheses:</strong></p>"
            for i, hypo in enumerate(hypotheses):
                html += f"""
                <div style="border-left: 3px solid #28a745; padding-left: 10px; margin: 10px 0;">
                    <h5>#{i + 1}: {hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p>{hypo.get("text", "No description")}</p>
                </div>
                """

        elif step_name in ["reflection", "reflection_evolved"]:
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Reviewed {len(hypotheses)} hypotheses:</strong></p>"
            for hypo in hypotheses:
                html += f"""
                <div style="border-left: 3px solid #17a2b8; padding-left: 10px; margin: 10px 0;">
                    <h5>{hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                       <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                    {f"<p><strong>Comments:</strong> {hypo.get('comments', 'No comments')}</p>" if hypo.get("comments") else ""}
                </div>
                """

        elif step_name.startswith("ranking"):
            hypotheses = step_data.get("hypotheses", [])
            if hypotheses:
                # Sort by Elo score
                sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("elo_score", 0), reverse=True)
                html += f"<p><strong>Ranking results ({len(hypotheses)} hypotheses):</strong></p>"
                html += "<ol>"
                for hypo in sorted_hypotheses:
                    html += f"""
                    <li style="margin: 5px 0;">
                        <strong>{hypo.get("title", "Untitled")}</strong> (ID: {hypo.get("id", "Unknown")}) 
                        - Elo: {hypo.get("elo_score", 0):.2f}
                    </li>
                    """
                html += "</ol>"

        elif step_name == "evolution":
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Evolved {len(hypotheses)} new hypotheses by combining top performers:</strong></p>"
            for hypo in hypotheses:
                html += f"""
                <div style="border-left: 3px solid #ffc107; padding-left: 10px; margin: 10px 0;">
                    <h5>{hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p>{hypo.get("text", "No description")}</p>
                </div>
                """

        elif step_name == "proximity":
            adjacency_graph = step_data.get("adjacency_graph", {})
            nodes = step_data.get("nodes", [])
            edges = step_data.get("edges", [])

            # Debug logging
            logger.info(
                f"Proximity data - adjacency_graph keys: {list(adjacency_graph.keys()) if adjacency_graph else 'None'}"
            )
            logger.info(f"Proximity data - nodes count: {len(nodes) if nodes else 0}")
            logger.info(f"Proximity data - edges count: {len(edges) if edges else 0}")

            if adjacency_graph:
                num_hypotheses = len(adjacency_graph)
                html += "<p><strong>Similarity Analysis:</strong></p>"
                html += f"<p>Analyzed relationships between {num_hypotheses} hypotheses</p>"

                # Calculate and display average similarity
                all_similarities = []
                for hypo_id, connections in adjacency_graph.items():
                    for conn in connections:
                        all_similarities.append(conn.get("similarity", 0))

                if all_similarities:
                    avg_sim = sum(all_similarities) / len(all_similarities)
                    html += f"<p>Average similarity: {avg_sim:.3f}</p>"
                    html += f"<p>Total connections analyzed: {len(all_similarities)}</p>"

                # Show top similar pairs
                similarity_pairs = []
                for hypo_id, connections in adjacency_graph.items():
                    for conn in connections:
                        similarity_pairs.append((hypo_id, conn.get("other_id"), conn.get("similarity", 0)))

                # Sort by similarity and show top 5
                similarity_pairs.sort(key=lambda x: x[2], reverse=True)
                if similarity_pairs:
                    html += "<h6>Top Similar Hypothesis Pairs:</h6><ul>"
                    for i, (id1, id2, sim) in enumerate(similarity_pairs[:5]):
                        html += f"<li>{id1} ↔ {id2}: {sim:.3f}</li>"
                    html += "</ul>"
                else:
                    html += "<p>No proximity data available.</p>"

        elif step_name == "meta_review":
            # Debug: log the actual meta_review data structure
            import sys

            print("DEBUG: meta_review step_data =", step_data, file=sys.stderr)
            assert isinstance(step_data, dict), "meta_review step_data is not a dict"
            # Accept both direct dict or nested under 'meta_review'
            if "meta_review" in step_data and isinstance(step_data["meta_review"], dict):
                meta_review = step_data["meta_review"]
            else:
                meta_review = step_data
            assert "meta_review_critique" in meta_review, f"meta_review_critique missing in meta_review: {meta_review}"
            assert "research_overview" in meta_review, f"research_overview missing in meta_review: {meta_review}"
            # Critique section
            if meta_review.get("meta_review_critique"):
                html += "<h5>Critique:</h5><ul>"
                for critique in meta_review["meta_review_critique"]:
                    html += f"<li>{critique}</li>"
                html += "</ul>"
            # Top ranked hypotheses section
            top_hypos = meta_review.get("research_overview", {}).get("top_ranked_hypotheses", [])
            assert isinstance(top_hypos, list), f"top_ranked_hypotheses is not a list: {top_hypos}"
            if top_hypos:
                html += "<h5>Top Ranked Hypotheses:</h5>"
                for i, hypo in enumerate(top_hypos):
                    html += f"""
                    <div style="border-left: 3px solid #28a745; padding-left: 10px; margin: 10px 0;">
                        <h6>#{i + 1}: {hypo.get("title", "Untitled")}</h6>
                        <p><strong>ID:</strong> {hypo.get("id", "Unknown")} | 
                           <strong>Elo Score:</strong> {hypo.get("elo_score", 0):.2f}</p>
                        <p><strong>Description:</strong> {hypo.get("text", "No description")}</p>
                        <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                           <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                    </div>
                    """
            # Suggested next steps section
            if meta_review.get("research_overview", {}).get("suggested_next_steps"):
                html += "<h5>Suggested Next Steps:</h5><ul>"
                for step in meta_review["research_overview"]["suggested_next_steps"]:
                    html += f"<li>{step}</li>"
                html += "</ul>"

        # Add timing information if available
        if step_data.get("duration"):
            html += f"<p><em>Duration: {step_data['duration']:.2f}s</em></p>"

        html += "</div></details>"

    # Final summary section - always expanded
    # Prefer ranking steps, else fallback to step with most hypotheses
    final_hypotheses = []
    final_step = None
    step_order = ["ranking_final", "ranking2", "ranking", "ranking1"]
    for step_name in step_order:
        if step_name in steps and steps[step_name].get("hypotheses"):
            final_hypotheses = steps[step_name]["hypotheses"]
            final_step = step_name
            break

    # Fallback: use step with most hypotheses if no ranking step exists
    if not final_hypotheses:
        max_count = 0
        for sname, sdata in steps.items():
            hypos = sdata.get("hypotheses", [])
            if len(hypos) > max_count:
                final_hypotheses = hypos
                final_step = sname
                max_count = len(hypos)

    # Assertions: final list should not be empty and no duplicate IDs (only for ranking steps)
    ranking_steps = ["ranking_final", "ranking2", "ranking", "ranking1"]
    if final_hypotheses:
        ids = [h.get("id") for h in final_hypotheses]
        if final_step in ranking_steps:
            assert len(ids) == len(set(ids)), "Duplicate hypothesis IDs found in final rankings!"
        assert len(final_hypotheses) > 0, "Final hypothesis list is empty!"

        # Sort by Elo score if present, else by ID
        if any("elo_score" in h for h in final_hypotheses):
            final_hypotheses = sorted(final_hypotheses, key=lambda h: h.get("elo_score", 0), reverse=True)
        else:
            final_hypotheses = sorted(final_hypotheses, key=lambda h: h.get("id", ""))

        html += """
        <div style="margin: 20px 0; padding: 15px; border: 2px solid #28a745; border-radius: 8px; background-color: rgba(40, 167, 69, 0.12); color: inherit;">
            <h3>🏆 Final Rankings - Top Hypotheses</h3>
        """
        if final_step not in ranking_steps:
            html += '<p style="color: #e67e22;">Warning: No ranking step found. Showing hypotheses from the latest available step ("{}"). These may not be ranked.</p>'.format(
                final_step
            )

        # Log final rankings if log_file is provided
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"--- Final Rankings Section (step: {final_step}) ---\n")
                for i, hypo in enumerate(final_hypotheses[:10]):
                    f.write(
                        f"  #{i + 1}: ID: {hypo.get('id')} | Title: {hypo.get('title')} | Elo: {hypo.get('elo_score', 'N/A')}\n"
                    )

        for i, hypo in enumerate(final_hypotheses[:10]):  # Show top 10
            rank_color = "#28a745" if i < 3 else "#17a2b8" if i < 6 else "#6c757d"
            html += f"""
            <div style="border-left: 4px solid {rank_color}; padding: 15px; margin: 10px 0; background-color: rgba(127, 127, 127, 0.10); color: inherit; border-radius: 5px;">
                <h4>#{i + 1}: {hypo.get("title", "Untitled")}</h4>
                <p><strong>ID:</strong> {hypo.get("id", "Unknown")} | 
                   <strong>Elo Score:</strong> {hypo.get("elo_score", 0):.2f}</p>
                <p><strong>Description:</strong> {hypo.get("text", "No description")}</p>
                <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                   <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                {data_check_html(hypo["data_check"]) if "data_check" in hypo else ""}
            </div>
            """

        html += "</div>"
    else:
        if errors:
            cause = "; ".join(sorted({classify_llm_error(e) for e in errors}))
            no_rank_msg = (
                f"No hypotheses available for final ranking because generation failed: {html_lib.escape(cause)}. "
                "See the details above."
            )
        else:
            no_rank_msg = "No hypotheses available for final ranking. This may indicate an error in the workflow."
        html += f"""
        <div style="margin: 20px 0; padding: 15px; border: 2px solid #e74c3c; border-radius: 8px; background-color: rgba(231, 76, 60, 0.12); color: inherit;">
            <h3>🏆 Final Rankings - Top Hypotheses</h3>
            <p style="color: #e74c3c;">{no_rank_msg}</p>
        </div>
        """
        # Log missing final rankings if log_file is provided
        if log_file:
            with open(log_file, "a") as f:
                f.write("--- Final Rankings Section: No hypotheses available for final ranking. ---\n")

    return html


def get_references_html(cycle_details: Dict, research_goal: Optional[ResearchGoal] = None) -> str:
    """Get references HTML for the cycle."""
    try:
        # Search for arXiv papers related to the research goal
        goal = research_goal or current_research_goal
        if goal and goal.description:
            arxiv_tool = ArxivSearchTool(max_results=5)
            papers = arxiv_tool.search_papers(query=goal.description, max_results=5, sort_by="relevance")

            if papers:
                html = "<h3>📚 Related arXiv Papers</h3>"
                for paper in papers:
                    html += f"""
                    <div style="border: 1px solid rgba(127, 127, 127, 0.35); padding: 15px; margin: 10px 0; border-radius: 8px; background-color: rgba(127, 127, 127, 0.08); color: inherit;">
                        <h4>{paper.get("title", "Untitled")}</h4>
                        <p><strong>Authors:</strong> {", ".join(paper.get("authors", [])[:5])}</p>
                        <p><strong>arXiv ID:</strong> {paper.get("arxiv_id", "Unknown")} | 
                           <strong>Published:</strong> {paper.get("published", "Unknown")}</p>
                        <p><strong>Abstract:</strong> {paper.get("abstract", "No abstract")[:300]}...</p>
                        <p>
                            <a href="{paper.get("arxiv_url", "#")}" target="_blank">📄 View on arXiv</a> | 
                            <a href="{paper.get("pdf_url", "#")}" target="_blank">📁 Download PDF</a>
                        </p>
                    </div>
                    """
                return html
            else:
                return "<p>No related arXiv papers found.</p>"
        else:
            return "<p>No research goal set for reference search.</p>"

    except Exception as e:
        logger.error(f"Error fetching references: {e}")
        return f"<p>Error loading references: {str(e)}</p>"


def create_gradio_interface():
    """Create the Gradio interface."""

    # Fetch models on startup
    fetch_available_models()

    # Get deployment status
    status_text, status_color = get_deployment_status()

    with gr.Blocks(
        title="Co-Scientist Swarm",
        theme=gr.themes.Soft(),
        css="""
        .status-box {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .status-box { color: inherit; }
        .orange { background-color: rgba(255, 193, 7, 0.16); border: 1px solid rgba(255, 193, 7, 0.55); }
        .blue { background-color: rgba(23, 162, 184, 0.14); border: 1px solid rgba(23, 162, 184, 0.5); }
        #results-panel pre, #references-panel pre { color: inherit; background: rgba(127, 127, 127, 0.10); }
        """,
    ) as demo:
        # Header
        gr.Markdown("# 🐝 Co-Scientist Swarm")
        gr.Markdown(
            "Generate research hypotheses, rank them by debate between model judges, and export every ranked idea "
            "with its review so it can be checked and tested. By [Saleem Al Dajani](https://github.com/saleemaldajani)."
        )

        # Deployment status
        gr.HTML(f'<div class="status-box {status_color}">🔧 Deployment Status: {status_text}</div>')

        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                # Research goal input
                research_goal_input = gr.Textbox(
                    label="Research Goal",
                    placeholder="Enter your research goal (e.g., 'Develop new methods for increasing the efficiency of solar panels')",
                    lines=3,
                )
                with gr.Row():
                    goal_file = gr.File(
                        label="Load the goal from a file (.md, .txt)",
                        file_types=[".md", ".txt"],
                        type="filepath",
                    )
                    data_files = gr.File(
                        label="Data files (optional: .csv, .tsv, .md, .txt, .json)",
                        file_count="multiple",
                        file_types=[".csv", ".tsv", ".md", ".txt", ".json"],
                        type="filepath",
                    )
                data_status = gr.Markdown(
                    "No data attached: the model sees the goal text only. Attach tables (and a README) to ground the "
                    "hypotheses in them: each step gets the rows most relevant to it, and every quoted number is checked "
                    "against the files. **Attached data is sent to the model provider you select**, so attach only what "
                    "you may share with it."
                )

                # Science tools (off by default: No tools sends the same prompts as before)
                with gr.Accordion("🧪 Science tools", open=False):
                    tools_mode = gr.Radio(
                        choices=TOOL_MODES,
                        value="No tools",
                        label="Tools",
                        info="Before writing hypotheses, the model can query scientific databases and literature "
                        "(UniProt, PubMed, ChEMBL, Ensembl, ClinVar, ClinicalTrials.gov and more) and web search. "
                        "Each step with tools makes two or three extra model calls (choose tools, call them, follow up).",
                    )
                    tools_pick = gr.CheckboxGroup(
                        choices=[("Web search", WEB_SEARCH_TOOL)] + tool_choices(),
                        value=[],
                        label="Tools to enable",
                        visible=False,
                    )

                # Simulation check (off by default): the top ideas are tested with simulation tools over MCP
                with gr.Accordion("🔬 Simulation check", open=False):
                    sim_mode = gr.Radio(
                        choices=server_choices(),
                        value="No simulation",
                        label="Simulation server",
                        info="After ranking, each of the top 3 ideas gets a simulation question; the model calls the "
                        "server's tools, reads the computed results, gives a verdict (supported, inconclusive or refuted) "
                        "and rewrites the idea so its numbers agree with the results. The records lab computes from the "
                        "attached tables; the team lab simulates specialist and generalist teams. About two extra model "
                        "calls per checked idea.",
                    )

                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    default_model = get_default_model_choice()
                    model_dropdown = gr.Dropdown(
                        choices=get_model_dropdown_choices(),
                        value=default_model,
                        label=f"LLM Model (default: {default_model})",
                        info="Compact free models are recommended first so demo runs finish faster.",
                    )

                    with gr.Row():
                        num_hypotheses = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=config.get("num_hypotheses", 4),
                            step=1,
                            label="Hypotheses per Cycle",
                        )
                        top_k_hypotheses = gr.Slider(minimum=2, maximum=5, value=2, step=1, label="Top K for Evolution")

                    with gr.Row():
                        generation_temp = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Generation Temperature (Creativity)"
                        )
                        reflection_temp = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Reflection Temperature (Analysis)"
                        )

                    elo_k_factor = gr.Slider(
                        minimum=1, maximum=100, value=32, step=1, label="Elo K-Factor (Ranking Sensitivity)"
                    )

                mode_radio = gr.Radio(
                    choices=["Single co-scientist", "Swarm", *TEAM_MODES],
                    value="Single co-scientist",
                    label="Mode",
                    info="Swarm: an orchestrator splits the goal, parallel workers generate, an LLM judge ranks by pairwise debate, "
                    "the top ideas are rewritten from their critiques. Research group, Startup team and Peer review give the "
                    "swarm's workers roles (settings below).",
                )
                with gr.Accordion("🐝 Swarm Settings", open=False, visible=False) as swarm_box:
                    with gr.Row():
                        swarm_workers = gr.Slider(minimum=2, maximum=6, value=3, step=1, label="Workers")
                        swarm_per_worker = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1, label="Hypotheses per Worker"
                        )
                    with gr.Row():
                        swarm_rounds = gr.Slider(
                            minimum=1, maximum=3, value=1, step=1, label="Rounds (evolution between rounds)"
                        )
                        swarm_matches = gr.Slider(
                            minimum=1, maximum=4, value=2, step=1, label="Debate Rounds per Round"
                        )
                role_models = [SAME_MODEL, *get_model_dropdown_choices()]
                with gr.Accordion("🎓 Research group settings", open=True, visible=False) as lab_box:
                    gr.Markdown(
                        "PIs set agendas and brief their students; students propose, discuss them at a lab meeting and "
                        "revise; each PI puts its best ideas forward, and the PIs judge them in debates. With more PIs "
                        "than students, two or four PIs co-advise each student. The two numbers must divide one another."
                    )
                    with gr.Row():
                        lab_pis = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="PIs")
                        lab_students = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Students")
                        lab_total = gr.Slider(minimum=4, maximum=12, value=8, step=1, label="Ideas in all")
                    with gr.Row():
                        lab_pi_model = gr.Dropdown(choices=role_models, value=SAME_MODEL, label="PI model")
                        lab_student_model = gr.Dropdown(choices=role_models, value=SAME_MODEL, label="Student model")
                with gr.Accordion("🚀 Startup team settings", open=True, visible=False) as biz_box:
                    gr.Markdown(
                        "The same structure with startup roles: co-founders each lead a team of junior members and "
                        "decide which ventures go forward. With no junior members, co-founders propose and review one "
                        "another's ideas. Works best with a business brief: a customer, a market size and a test that "
                        "could prove the idea wrong."
                    )
                    with gr.Row():
                        biz_founders = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Co-founders")
                        biz_juniors = gr.Slider(minimum=0, maximum=8, value=0, step=1, label="Junior members")
                        biz_total = gr.Slider(minimum=4, maximum=12, value=8, step=1, label="Ideas in all")
                with gr.Accordion("📝 Peer review settings", open=True, visible=False) as rev_box:
                    gr.Markdown(
                        "Authors submit ideas; every reviewer referees every submission blind (score 1 to 10 and a "
                        "recommendation); authors revise and reply; reviewers score the revisions, and the final list "
                        "follows the second scores."
                    )
                    with gr.Row():
                        rev_authors = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Authors")
                        rev_reviewers = gr.Slider(minimum=1, maximum=3, value=3, step=1, label="Reviewers")
                        rev_total = gr.Slider(minimum=4, maximum=12, value=8, step=1, label="Ideas in all")
                    rev_reviewer_model = gr.Dropdown(choices=role_models, value=SAME_MODEL, label="Reviewer model")

                export_checkbox = gr.Checkbox(
                    value=True,
                    label="Export results",
                    info="Download the ranked ideas as .html, .md and .json, with reviews and references.",
                )

                # Single action button
                with gr.Row():
                    run_cycle_btn = gr.Button("🔄 Run Cycle", variant="primary")

                # Status display
                status_output = gr.Textbox(
                    label="Status",
                    value="Enter a research goal and click 'Run Cycle' to begin.",
                    interactive=False,
                    lines=3,
                )

                bundle_file = gr.File(label="Results (.zip)", interactive=False)

            with gr.Column(scale=1):
                # Instructions
                gr.Markdown("""
                ### 📖 Instructions

                1. **Enter a research goal**: the question, the data you can use, and what each hypothesis should contain (or load it from a .md/.txt file).
                2. **Attach data** (optional): tables (.csv, .tsv) and a README; every step sees the parts relevant to it, and quoted numbers are checked against them.
                   **Science tools** (optional, off by default): let the model query databases, literature and the web before it writes hypotheses; choose all tools or pick them.
                3. **Choose a mode**: *Single co-scientist* (one generation call per cycle), *Swarm* (parallel workers ranked by debate), or an organized swarm: *Research group* (PIs and students), *Startup team* (co-founders and junior members) or *Peer review* (authors and reviewers).
                4. **Adjust settings** (optional): model, hypotheses per cycle, the swarm's workers, rounds and debates, and each team's size and role models.
                5. **Run**: results appear below; with **Export results** ticked, a zip of .html, .md and .json files is ready to download.

                ### 💡 Tips
                - Start small on the free models: 3 workers × 3 hypotheses, 1 round
                - Larger models give sharper debates; compact free models are listed first because they are faster
                - Each single-mode cycle builds on the previous one; a swarm or team run is self-contained
                - Science tools and the simulation check run in Single and Swarm modes; team modes use the goal and attached data
                - References are written by the model, not retrieved: check them before relying on them
                
                **Note:** Free models can be rate-limited or slow. The app will try a few free fallbacks automatically, and you can select a different recommended free model in Advanced Settings.
                """)

        with gr.Tabs():
            with gr.Tab("Current Run"):
                with gr.Row():
                    with gr.Column():
                        results_output = gr.HTML(
                            elem_id="results-panel",
                            label="Results",
                            value="<p>Results will appear here after running cycles.</p>",
                        )

                with gr.Row():
                    with gr.Column():
                        references_output = gr.HTML(
                            elem_id="references-panel",
                            label="References",
                            value="<p>Related research papers will appear here.</p>",
                        )

            with gr.Tab("Run History") as run_history_tab:
                gr.Markdown("Saved runs load automatically. Use refresh if runs were changed outside this page.")
                refresh_history_btn = gr.Button("Refresh History")
                history_output = gr.HTML(label="Saved Runs", value=history_html())
                with gr.Row():
                    delete_run_dropdown = gr.Dropdown(
                        choices=history_run_choices(),
                        label="Saved Run to Delete",
                        interactive=True,
                    )
                    delete_history_btn = gr.Button("Delete Selected Run", variant="stop")
                delete_history_status = gr.Markdown()

        # Event handler: single button sets research goal and runs cycle
        def run_full_cycle(
            research_goal,
            llm_model,
            num_hypotheses,
            generation_temp,
            reflection_temp,
            elo_k_factor,
            top_k_hypotheses,
            mode="Single co-scientist",
            workers=3,
            per_worker=3,
            rounds=1,
            matches=2,
            export=True,
            data_paths=None,
            tools_mode="No tools",
            tools_pick=None,
            sim_mode="No simulation",
            lab_pis=1,
            lab_students=4,
            lab_total=8,
            lab_pi_model=SAME_MODEL,
            lab_student_model=SAME_MODEL,
            biz_founders=2,
            biz_juniors=0,
            biz_total=8,
            rev_authors=4,
            rev_reviewers=3,
            rev_total=8,
            rev_reviewer_model=SAME_MODEL,
        ):
            data = DataBundle.from_paths(data_paths or []) or None
            if mode in TEAM_MODES:
                yield from run_team(
                    mode,
                    research_goal,
                    llm_model,
                    export,
                    data,
                    dict(
                        lab_pis=lab_pis,
                        lab_students=lab_students,
                        lab_total=lab_total,
                        lab_pi_model=lab_pi_model,
                        lab_student_model=lab_student_model,
                        biz_founders=biz_founders,
                        biz_juniors=biz_juniors,
                        biz_total=biz_total,
                        rev_authors=rev_authors,
                        rev_reviewers=rev_reviewers,
                        rev_total=rev_total,
                        rev_reviewer_model=rev_reviewer_model,
                    ),
                    skipped=tools_mode != "No tools" or sim_mode != "No simulation",
                )
                return
            tools, web = build_tools(tools_mode, tools_pick)
            try:
                sim = build_simulation(sim_mode, data_paths)
            except ValueError as e:
                yield (f"❌ {e}", "", "", history_html(), gr.update(choices=history_run_choices(), value=None), None)
                return
            if mode == "Swarm":
                if not (research_goal or "").strip():
                    yield (
                        "❌ Error: Please enter a research goal.",
                        "",
                        "",
                        history_html(),
                        gr.update(choices=history_run_choices(), value=None),
                        None,
                    )
                    return
                for status, results, references in run_swarm_with_progress(
                    research_goal.strip(),
                    llm_model,
                    workers,
                    per_worker,
                    rounds,
                    matches,
                    data=data,
                    tools=tools,
                    web=web,
                    sim=sim,
                ):
                    yield (
                        status,
                        results,
                        references,
                        history_html(),
                        gr.update(choices=history_run_choices(), value=None),
                        None,
                    )
                if export:
                    yield (
                        status,
                        results,
                        references,
                        history_html(),
                        gr.update(choices=history_run_choices(), value=None),
                        export_bundle_zip("Swarm"),
                    )
                return
            # Set research goal
            status_msg, _ = set_research_goal(
                research_goal,
                llm_model,
                num_hypotheses,
                generation_temp,
                reflection_temp,
                elo_k_factor,
                top_k_hypotheses,
                data=data,
                tools=tools,
                web=web,
            )
            yield (
                f"{status_msg}\n\nStarting cycle with a {format_timeout_duration(CYCLE_TIMEOUT_SECONDS)} limit.",
                "<p>Starting cycle...</p>",
                "",
                history_html(),
                gr.update(choices=history_run_choices(), value=None),
                None,
            )
            for status, results, references in run_cycle_with_progress():
                yield (
                    f"{status_msg}\n\n{status}",
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
                    None,
                )
            if sim and current_research_goal is not None and global_context.hypotheses:
                yield (
                    f"{status_msg}\n\n{status}\n\n🔬 Running the simulation check on the top ideas...",
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
                    None,
                )
                try:
                    results = (results or "") + simulation_check_single(sim, current_research_goal, global_context)
                except Exception as e:  # the ranked ideas stay; report why the check did not run
                    results = (results or "") + f"<p>⚠️ Simulation check failed: {e}</p>"
                yield (
                    f"{status_msg}\n\n{status}",
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
                    None,
                )
            if export:
                yield (
                    f"{status_msg}\n\n{status}",
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
                    export_bundle_zip("Single co-scientist"),
                )

        def run_team(mode, research_goal, llm_model, export, data, team, skipped=False):
            def out(status, results="", references="", bundle=None):
                return (
                    status,
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
                    bundle,
                )

            if not (research_goal or "").strip():
                yield out("❌ Error: Please enter a research goal.")
                return
            model = llm_model if llm_model and llm_model != "-- Select Model --" else None
            log: List[str] = []
            try:
                swarm, organization = build_team(
                    mode, research_goal.strip(), model, data=data, progress=log.append, **team
                )
            except ValueError as e:
                yield out(f"❌ {e}")
                return
            note = (
                "Science tools and the simulation check run in Single and Swarm modes only; this run uses neither.\n"
                if skipped
                else ""
            )
            status = results = references = ""
            for status, results, references in stream_swarm_run(
                swarm, research_goal.strip(), model, log, TEAM_TIMEOUT_SECONDS, label=mode, organization=organization
            ):
                yield out(note + status, results, references)
            if export and LAST_SWARM.get("mode") == mode.lower():
                yield out(note + status, results, references, export_bundle_zip("Swarm"))

        run_cycle_btn.click(
            fn=run_full_cycle,
            inputs=[
                research_goal_input,
                model_dropdown,
                num_hypotheses,
                generation_temp,
                reflection_temp,
                elo_k_factor,
                top_k_hypotheses,
                mode_radio,
                swarm_workers,
                swarm_per_worker,
                swarm_rounds,
                swarm_matches,
                export_checkbox,
                data_files,
                tools_mode,
                tools_pick,
                sim_mode,
                lab_pis,
                lab_students,
                lab_total,
                lab_pi_model,
                lab_student_model,
                biz_founders,
                biz_juniors,
                biz_total,
                rev_authors,
                rev_reviewers,
                rev_total,
                rev_reviewer_model,
            ],
            outputs=[
                status_output,
                results_output,
                references_output,
                history_output,
                delete_run_dropdown,
                bundle_file,
            ],
        )

        def load_goal_file(path):
            if not path:
                return gr.update()
            try:
                return open(path, encoding="utf-8", errors="replace").read()
            except OSError as e:
                return f"Could not read the goal file: {e}"

        def describe_data(paths):
            if not paths:
                return "No data attached: the model sees the goal text only."
            d = DataBundle.from_paths(paths)
            if not d:
                return "No usable data files. " + "; ".join(d.skipped)
            return (
                f"**Data attached:** {d.summary()}. Each step gets an extract of about {20000:,} characters "
                "(small tables whole, the most relevant rows of large ones); every quoted number is checked against these files. "
                "**Attached data is sent to the model provider you select.**"
            )

        goal_file.upload(fn=load_goal_file, inputs=[goal_file], outputs=[research_goal_input])
        data_files.change(fn=describe_data, inputs=[data_files], outputs=[data_status])
        tools_mode.change(
            fn=lambda m: gr.update(visible=m == "Choose tools"), inputs=[tools_mode], outputs=[tools_pick]
        )
        mode_radio.change(fn=mode_panels, inputs=[mode_radio], outputs=[swarm_box, lab_box, biz_box, rev_box])

        demo.load(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, delete_history_status],
        )
        run_history_tab.select(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, delete_history_status],
        )
        refresh_history_btn.click(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, delete_history_status],
        )
        delete_history_btn.click(
            fn=delete_history_run,
            inputs=[delete_run_dropdown],
            outputs=[delete_history_status, history_output, delete_run_dropdown],
        )

        # Example inputs
        gr.Examples(
            examples=[
                ["Develop new methods for increasing the efficiency of solar panels"],
                ["Create novel approaches to treat Alzheimer's disease"],
                ["Design sustainable materials for construction"],
                ["Improve machine learning model interpretability"],
                ["Develop new quantum computing algorithms"],
                [
                    "Identify wearable signals (heart rate, sleep, activity) that change months before a new chronic disease diagnosis"
                ],
                [
                    "Propose testable hypotheses that combine wearable data with blood biomarkers to measure biological aging"
                ],
                [
                    "Find interventions whose effect on resting heart rate and sleep could forecast their effect on long-term health"
                ],
                [
                    "Design ways to tell a sustained physiological change from a sensor artifact in consumer wearable data"
                ],
                ["Propose multimodal hypotheses linking proteomics and lifestyle surveys to healthy aging"],
            ],
            inputs=[research_goal_input],
            label="Example Research Goals",
        )

        # GitHub icon and link at the bottom
        gr.HTML(
            """
            <div style="text-align:center; margin-top: 30px;">
                <a href="https://github.com/saleemaldajani/co-scientist" target="_blank" style="text-decoration:none; display:inline-flex; align-items:center; gap:8px;">
                    <svg height="32" width="32" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:middle;">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
                        -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2
                        -3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64
                        -.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08
                        2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                        1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    <span style="font-size: 1.1em; vertical-align:middle;">View on GitHub</span>
                </a>
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Warning: OPENROUTER_API_KEY environment variable not set.")
        print("The app will start but may not function properly without an API key.")

    # Create and launch the Gradio app
    demo = create_gradio_interface()

    # Launch with appropriate settings for HF Spaces
    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[str(reports_dir.resolve())],
    )
