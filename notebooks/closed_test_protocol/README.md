# Closed-test protocol (new folder)

This folder does not replace or modify the existing notebooks. The published 70/15/15 chronological split and `CURRENT.json` stay as they are. The source data and `process_data.py` do not need to be rerun.

## Current recorded state (2 November 2026)

The development decision is frozen in `results/frozen_decision.json`. The final evaluation was then executed once; its aggregate scores and provenance are in `results/final_test_metrics.json`. The executed final notebook is preserved in `results/executed_notebook_snapshots/08_final_test_once_executed.ipynb`. The main `08_final_test_once.ipynb` retains the corrected reusable code with `RUN_FINAL_TEST = False` and no saved outputs. `09_final_test_figures.ipynb` contains the resulting figures.

The row-level prediction CSV files remain local and are excluded from Git. They can be regenerated only by running the final notebook with the original H&M data, which are not redistributed here. The saved notebook outputs and aggregate metrics remain viewable on GitHub. Earlier work on this same publication had already inspected test outcomes, so this protocol records a fresh development sequence but cannot make the test historically unseen.

## Run order

1. Run `03_development_regression_eda.ipynb` and `04_development_classification_eda.ipynb`. These use train and validation only. They provide a compact development-only replacement for the test-dependent parts of the old 03 and 04; the old EDA remains archived.
2. Existing notebooks 05 and 06 already use train and validation. Their saved results can be kept; rerun them only if their own code or input publication changes. Notebook 08 also uses development rows only.
3. Run `07_development_only.ipynb` once, from a fresh kernel. It reruns nested regression, classification and threshold selection. This is the costly step. Its final cell freezes decisions in `results/frozen_decision.json`. It reads no test rows and saves development-only diagnostics under this folder.
4. Inspect development results, then stop changing models, variables, thresholds and metrics. Only then open `08_final_test_once.ipynb`, set `RUN_FINAL_TEST = True`, and run all its cells once. It refits the chosen models on every train + validation row and evaluates every test row. It refuses to overwrite an existing final result.

Creating these files did not run any models; the executions recorded above happened later. There is no need to rerun original 01 or 02 when retaining the current publication. The 07 development notebook and final test notebook both fit models; the final notebook does **not** repeat nested search.

The previous test outcomes were viewed in earlier work. This folder enforces a closed test **from this new run onward** and records that historical limitation. A new notebook cannot make the old test historically unseen.

5. Run 09_final_test_figures.ipynb after the final test. It reads saved predictions and exports the two report figures as PDF and PNG to the imagenes folder without refitting.

For the final test already executed on 2 November 2026 before prediction export was added: keep its kernel alive and run a new code cell containing: exec((ROOT / 'notebooks/closed_test_protocol/save_predictions_current_kernel.py').read_text(encoding='utf-8')). This only saves predictions already in memory; then run 09_final_test_figures.ipynb. Do not rerun model fitting.
