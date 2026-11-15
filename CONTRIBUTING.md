# Contributing: human and AI collaborators

- Reuse existing PerceptionLab nodes before writing new ones.
- Each experiment needs a falsifiable question, host-loadable saved graph, a baseline, runnable tests, numbers, provenance, and limitations.
- Preserve failed runs. Mark hypotheses as hypotheses; do not rename a display to make a claim true.
- Verify every wiring port and the *effect* of every alleged control.
- Keep experimental scaffolding here. Propose only reusable nodes and tested workflows upstream through review.
- Use branches and PRs when working together. Do not overwrite another contributor's changes.
- Credit code and scientific sources. No unreviewed changes to PerceptionLab main.
- No hype, no invented measurements, no quantum/biological claims from analogies.

Run `python -m unittest discover -s tests -v` and `python experiments/001_causal_age/run.py --upstream PerceptionLab` with upstream checked out beside this repo, or use the included GitHub Actions workflow.

## Keep the AI workshop manuals up to date

Every new/revised node must have its class, actual port names, behaviour,
limits and upstream status in [Node Manual](docs/NODE_MANUAL.md). Every
numbered and control workflow must appear in
[Workflow Manual](docs/WORKFLOW_MANUAL.md), with a tested JSON graph,
dossier and durable result receipt. AI work requested by the owner
goes to **main** by default for normal `git pull`, not a new branch
that hides the files. Keep failures and null results visible.

