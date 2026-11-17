# demo_MMM fix report (Nazia131/demo_MMM, 1 commit, ~12.7k LOC py)
Checks run: compileall (clean), ruff F/E9/B/PLE, pytest (29 pass, 2 skip: torch absent), targeted greps.
NOT verified: training/inference runtime (torch not installed), API endpoints, model quality.
Overall health ~85/100. Report accuracy ~90% (static findings 85-99%, noted per item).

## Instructions for fixer (do in order; no exploring needed)

1. [HIGH, 99%] evaluate/zero_shot_eval.py:19
   `from utils.config import get_harm_categories` -> add `get_coefficient`.
   Used at L319 and L746 (undefined => NameError). L319 sits in a try block, so it likely falls
   back silently to a default threshold (85%): check the except branch after fixing.

2. [HIGH, 90%] model/apply_training_fixes.py is an unapplied one-off patch script.
   Evidence: model/gat_model.py:59 still `GAT_DROPOUT: float = 0.6` (script sets 0.3).
   Run from repo root: `python model/apply_training_fixes.py` (script uses model/ paths).
   If it aborts on a missing target line, apply the edits by hand: GAT_DROPOUT 0.3, _LEARNING_RATE
   0.001 in model/train.py, best-F1 checkpoint logic. Then `git rm` the script and any *.bak files.
   (train.py LR not checked, 60%.)

3. [MED, 95%] Large binaries committed: model/cache/graphs_*.pt (~78 MB + 2.6 MB).
   `git rm --cached -r model/cache` and add `model/cache/` to .gitignore.
   Keep model/checkpoints/best_gat.threshold.json (intentional; .pt checkpoints already ignored).

4. [MED, 90%] api/app.py:696 CORS `allow_origins=["*"]` with `allow_credentials=True`.
   Fix: read origins from env var (default localhost) or set allow_credentials=False.

5. [LOW, 99%] Cleanup:
   - tests/test_policy_consistency.py:21 remove unused import EDGE_TYPES.
   - harm_pattern_library/subgraph_miner.py:190 rename unused loop var `key` -> `_`.

6. [LOW, 95%] zip() without strict=: encoders/image_encoder.py:207, encoders/text_encoder.py:153,
   encoders/video_encoder.py:287, explainability/report_generator.py:101, graph/graph_builder.py:343.
   Add `strict=True` only where lengths must match (Python 3.10+ ok per requirements).

7. [LOW, 70%] data/download_hm.py (23 lines) overlaps data/download_datasets.py; remove if redundant.

## Ignore (false positives)
- api/app.py:794-802 B008 `File()` in defaults: standard FastAPI idiom.
- README "missing files" are bare filenames, not real breakage.

## Gaps / suggestions (optional)
- Tests don't cover api/, model/, graph/, evaluate/. Add smoke tests if time allows.
- Saved threshold metrics (thr 0.30): F1 0.566, AUC 0.708, acc 0.603: weak; fixes in #2 may help.
- requirements.txt: `clip @ git+...` unpinned; pin a commit for reproducibility.

## Verify after edits
`python -m compileall -q . && ruff check --select F,E9 . && pytest -q tests`
