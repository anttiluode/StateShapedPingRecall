# Repository repair and reproduction

15 November 2026.

The initial repository upload flattened the original archive layout. `python verify.py` failed with `FileNotFoundError` for `vendor/experiments/009_morphology_as_memory/core.py`; `analyze.py` likewise expected traces under `traces/`.

The directories were restored locally from the already committed files. While reproduction was running, Fable independently committed the same layout restoration in `5a4684baaf24f775c8a2a54d23884dc26fd51466`, together with the new damping experiment. That concurrent commit was retained; this review adds documentation rather than replacing Fable’s work. No simulation, scoring, node, protocol or original receipt contents were changed. The pinned runner and protocol hashes therefore remain valid. The copied workshop contribution instructions were replaced with instructions for this standalone repository.

Verification used Python 3.12.14, NumPy 2.3.5, opencv-python-headless 5.0.0.93 and Matplotlib 3.10.8. The original report used Python 3.12.

- `python verify.py`: passed exact 009 impulse equivalence for weights, branch survival and firing times, plus per-packet area and squared-drive checks.
- `python run.py --smoke --workers 1`: completed all eight development bodies. The archived smoke receipt comes from an earlier runner hash and differs in local-history values; it is retained as historical plumbing evidence, not a numerical regression target. The full scientific receipt matches the current runner hash and reproduces exactly.
- Full `python run.py --workers 4`: completed all 48 paired conditions / 96 bodies in approximately 77 seconds. Parsed `results.json` matched the committed receipt exactly after excluding only `elapsed_seconds`.
- `python inverse_control.py`: all 32 mixtures completed; parsed output exactly matched the committed inverse receipt.
- `python analyze.py`: completed, loaded the restored traces and regenerated the figure; parsed `summary.json` exactly matched the committed summary.

Runs were performed in a separate working copy. Original result receipts and the original figure are preserved. This is a reproduction with the same algorithm, seeds and dependencies, not an independent experiment or evidence of generalization.
