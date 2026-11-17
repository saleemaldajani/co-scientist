---
title: Co-Scientist Swarm
emoji: 🐝
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Generate, debate-rank and export research hypotheses
---

# Co-Scientist Swarm

**Saleem Al Dajani** · [GitHub](https://github.com/saleemaldajani) · [Hugging Face](https://huggingface.co/sdajani)

**Demo:** https://huggingface.co/spaces/sdajani/co-scientist-public-demo

Language models can propose research hypotheses faster than anyone can check them. The bottleneck is no longer generation but ranking and verification: deciding which hypotheses are worth testing, and checking that each one names real data and a test that can actually be run. I built this fork to study that bottleneck. It generates hypotheses for a research goal, ranks them by structured debate between model judges, and exports every ranked idea with its review so it can be audited and tested outside the app.

## What this fork adds

### Swarm mode
A single co-scientist asks one model call for all hypotheses at once, then ranks them by their review labels. Swarm mode instead divides the goal across several agents and ranks the pooled hypotheses by debate:

1. **Orchestrator.** Splits the goal into distinct research directions, each with an objective, the data types to favor, and boundaries.
2. **Workers.** One per direction, generating hypotheses in parallel.
3. **Proximity.** Merges near-duplicates by sentence-embedding similarity, so repeats do not dilute the ranking.
4. **Reflection.** Reviews each hypothesis for novelty and feasibility, with references.
5. **Tournament.** An LLM judge argues for and against each of two hypotheses, then picks a winner; Elo ratings update after every debate, with Swiss pairing by current rating and randomized order.
6. **Meta-review.** Reads the ranking, reviews and the judges' reasons, and writes guidance for improvement.
7. **Evolution.** Rewrites the top hypotheses from their critiques; the rewrites re-enter the tournament in the next round.

Every model call, debate and merge is logged with the run. The single co-scientist mode is unchanged, so the two can be compared on the same goal and model.

### Attach data
Models only read text, so the stock app saw nothing but the goal and invented its numbers. Attach tables
(`.csv`, `.tsv`) and text files (`.md`, `.txt`, `.json`, such as a README or data dictionary) next to the goal, or load
the goal itself from a file:

- **An extract per step.** Each prompt gets about 20,000 characters of the data: text files and small tables whole,
  and for large tables the rows most relevant to that step (the goal for the orchestrator, a worker's direction, the two
  ideas a judge compares, the idea being rewritten).
- **A check of every quoted number.** After the run, each number a hypothesis quotes is looked up in the files and shown
  as found, with its file and row, or not found. The check is in the results panel and in **Export results**.

Attached data is sent to the model provider you select; with a local model it stays on your machine.

### Science tools
Under **🧪 Science tools**, choose **No tools** (the default: prompts are exactly as before), **All tools**, or
**Choose tools** and tick the ones you want. Before each step that writes hypotheses (a single-mode generation; each
swarm worker), the model can then query 32 scientific databases and literature services, and the web:

- **Genes, variants and regulation:** Ensembl, ClinVar, dbSNP, gnomAD, GTEx, ENCODE cCREs, JASPAR, UniBind, UCSC
  conservation and binding sites, NCBI sequences.
- **Proteins and structures:** UniProt, InterPro, PDB, AlphaFold DB, Foldseek, STRING, Human Protein Atlas, QuickGO,
  Reactome, sequence alignment and similarity search.
- **Drugs, trials and targets:** ChEMBL, PubChem, openFDA, ClinicalTrials.gov, Open Targets, EMBL-EBI ontologies.
- **Literature:** PubMed, Europe PMC, OpenAlex, arXiv, bioRxiv; and **web search**.

With many tools on, the model first picks up to two that fit the goal (or the worker's direction), then reads their
command-line help and writes up to three calls. Each call runs one script from `app/science_skills/` with a 60-second
limit, in a scratch folder, without a shell, and without the model provider's key. The model then sees that output and
can make up to three follow-up calls, for example to fetch the abstracts of the articles a PubMed search found or the
records of the genes and compounds it named (a repeat of an earlier call is skipped). The output of both rounds (and
any file a call writes) goes into that step's prompt as a TOOL RESULTS block of about 12,000 characters (up to 6,000
per call; when it does not all fit, short outputs stay whole and the long ones share the rest). A step with tools makes two or three extra model calls. Every chosen tool,
call and output is kept with the run and exported as `share/tool_calls.json`.

Tool results come from public services and can be incomplete; each service has its own terms of use
(`app/science_skills/SKILL_LICENSES.md`). Some services allow more requests with a free key (`NCBI_API_KEY`,
`OPENALEX_API_KEY`, `FDA_API_KEY` in the environment).

### Simulation check
Under **🔬 Simulation check**, choose **No simulation** (the default: runs are exactly as before) or a simulation
server. After a run has ranked its ideas, each of the top three goes through four steps: the app lists the server's
tools over the Model Context Protocol (MCP); the model writes a simulation question that would test the idea and up to
three tool calls that answer it; the calls run in a separate server process; and the model reads the computed results,
gives a verdict (supported, inconclusive or refuted) with the numbers, and rewrites the idea so every number it states
agrees with the results. Checked ideas are listed by verdict, supported first. Two servers ship with the app
(`app/sim_servers/`):

- **Records lab** computes from the attached tables: rates and ratios by group, rate ratios with 95% intervals, and,
  for the London cholera records, a held-out test of the water-supplier explanation, an elevation test, the change
  between the 1849 and 1854 epidemics, and a simulation of the Broad Street outbreak without the handle removal.
- **Team lab** simulates teams of specialists and generalists completing multi-skill tasks (Monte Carlo), for goals
  about how to organize teams.

Any stdio MCP server can be used from Python with `SimulationLab(command=[...])`. Servers run without the model
provider's key. Each checked idea costs about two extra model calls.

### Export results
Tick **Export results** (on by default) to download a zip of each run:

| File | Contents |
|---|---|
| `share/ideas_bundle.json` | every ranked idea: text, Elo, review labels, references, debate record |
| `share/ideas_bundle.md`, `share/ideas_bundle.html` | a leaderboard, then each idea with its summary, full text, review and references |
| `share/idea_rankNN.html` | one page per idea, with navigation between ranks |
| `reviews/review_<id>.md` | the review of each idea, with its debates in swarm mode |
| `references.csv`, `references.md` | every reference, with the idea it belongs to |
| `share/tool_calls.json` | runs with science tools: each step's chosen tools, calls and output |

References are written by the model, not retrieved, so resolve each one before relying on it.

## Quick start

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key      # https://openrouter.ai/keys; free models need no credit
python app.py                           # http://localhost:7860
```

Enter a research goal, choose **Single co-scientist** or **Swarm** under Mode, adjust the swarm settings if needed, and run.

### Local open models
The app talks to any OpenAI-compatible endpoint, so it can run entirely on a local machine, for example with [Ollama](https://ollama.com):

```yaml
# config.yaml
openrouter_base_url: "http://localhost:11434/v1"
llm_model: "gpt-oss:120b"               # or gemma4:31b, ...
llm_request_timeout_seconds: 3600       # large local models are slow
```

Set `OPENROUTER_API_KEY` to any non-empty string (Ollama ignores it). For long research prompts, give the model a large enough context window, for example with an Ollama Modelfile containing `PARAMETER num_ctx 32768`.

### From Python

```python
from app.swarm import SwarmCoScientist
from app.export import ideas_from_hypotheses, write_bundle

from app.data_context import DataBundle

data = DataBundle.from_dir("my_tables/")  # optional
res = SwarmCoScientist(goal, model="gpt-oss:120b", n_workers=4, per_worker=10, rounds=2, data=data).run()
final = [h for h in res["hypotheses"] if h["id"] in res["final"]]
write_bundle(ideas_from_hypotheses(final, res["matches"]), "results/my_run", "my_run", goal, mode="swarm")
```

## Tests

```bash
make test     # offline: no key and no network; the model calls are mocked
make lint
```

## Credits and license

This is a fork of [llnl/open-ai-co-scientist](https://github.com/llnl/open-ai-co-scientist) (LLNL-CODE-2010270), whose single co-scientist cycle, interface and run history it builds on. Released under the MIT License (see `LICENSE` and `NOTICE`).

The science tools in `app/science_skills/` are copied unchanged from the open-source
[science-skills](https://github.com/google-deepmind/science-skills) collection: scripts under the Apache License 2.0,
other material under CC BY 4.0 (see `app/science_skills/README.md`).

The swarm design follows the tournament, evolution and meta-review of Co-Scientist (Gottweis et al., *Nature*, 2026), the routines and handoffs of OpenAI Swarm, orchestrator-worker research agents, and multi-agent debate (Du et al., 2023).
