# Science tools

Copied from the open-source science-skills collection (32 skills, commit `370f8f3`) by `scripts/update_science_skills.py`. The skills keep their original files and copyright notices, except that the scripts that tell a service who is calling (STRING, Europe PMC, OpenAlex) name this app; each of those files says so below its license header. Scripts and code are under the Apache License 2.0 (`LICENSE`); the other material is under CC BY 4.0. Each data source has its own terms of use (`SKILL_LICENSES.md`).

`catalog.json` is what the app reads: each skill's description, scripts, `--help` text and example commands.

Not copied:

- `alphagenome_atlas_website_links`: needs an AlphaGenome API key and the alphagenome, pandas packages
- `alphagenome_single_variant_analysis`: needs an AlphaGenome API key and the alphagenome, matplotlib, pandas packages
- `alphagenome_variant_impact_score`: needs an AlphaGenome API key and the alphagenome, anndata, polars, pysam packages
- `credentials`: instructions for storing API keys, no tool
- `predictingthepast`: downloads a model and needs JAX
- `pymol`: drives the PyMOL desktop program
- `uv`: instructions for installing the uv package manager, no tool
- `workflow_skill_creator`: instructions for writing new skills, no tool
