# Contributing

This is a standalone, headless recall assay derived from PerceptionLabAI 009. It does not contain the workshop's numbered experiment tree, GUI workflows or node manuals. Contributions to that broader workshop belong in [PerceptionLabAI](https://github.com/anttiluode/PerceptionLabAI).

- Preserve the original protocol, negative result, receipts and source pins. Put new experimental hypotheses in separate files and label exploratory choices.
- Verify actual wiring and timing. Reuse the vendored nodes for reproduction; do not silently substitute newer upstream versions.
- Run `python verify.py`, then `python run.py --smoke --workers 1`. Changes to simulation or scoring also require the full batch, inverse diagnostic and analysis described in the README, with an explanation of any changed results.
- Keep `vendor/` and `traces/` intact. Uploading all files into the repository root breaks imports and analysis.
- Separate a communication channel, a trained prediction rule and a supplied inverse model. No biological or consciousness claims follow from the analogy.
- Use branches/PRs for outside contributions. Antti's explicitly requested maintenance may go to main; fetch current main before publishing and preserve concurrent work.

The scripts overwrite matching output files. Run scientific reproduction in a disposable clone and compare it with the committed receipts before deciding whether a changed output should be published.
