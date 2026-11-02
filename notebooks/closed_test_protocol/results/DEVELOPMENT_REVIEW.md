# Development-only review

This report uses saved development outputs and VS Code recovery snapshots. No model was retrained and no final-test outcome was accessed during this review.

**Historical checkpoint:** this review was written before the final-test run. The test has since been executed; see `final_test_metrics.json` and the executed 08 snapshot. Statements below about the test still being closed describe the state at the time of this review. The source development notebooks now also contain their saved execution outputs.

## Completion and provenance

- Publication: 20261027T145552-5a1ac59a
- Dataset SHA-256: 7565450ec43df08610ee05892bbdb9931b27f389880e7677dc365d03af7b01f7
- Development EDA 03: 2/2 code cells executed, no saved errors.
- Development EDA 04: 2/2 code cells executed, no saved errors.
- Development notebook 07: 10/10 code cells executed, no saved errors.
- At this checkpoint, executed notebook snapshots were in `executed_notebook_snapshots/`; source notebook outputs were saved later.
- At this checkpoint, no final-test metrics file existed and the test guard remained closed.

## Regression: three later outer windows

| Fold | Selected candidate | Model MAE | Persistence MAE | Four-week mean MAE | Four-week median MAE |
|---:|---|---:|---:|---:|---:|
| 1 | boosting_absolute_error | 10.211 | 11.557 | 12.850 | 13.017 |
| 2 | boosting_absolute_error | 7.897 | 8.543 | 11.433 | 10.833 |
| 3 | boosting_absolute_error | 6.454 | 7.202 | 8.174 | 8.310 |
| **Mean** | | **8.187** | **9.101** | **10.819** | **10.720** |

All three windows selected absolute-error Boosting. MAE is in recorded next-week units per article-week; lower is better. These are development-window scores, not final-test scores.

## Classification and probability threshold

- Inner average precision selected `logistic_C_1` as the final classifier.
- Mean outer PR-AUC: **0.877** for the selected classifier and **0.689** for the current-sales rule.
- The inner-selected thresholds in the three outer folds were **0.34, 0.42, 0.29**.
- A separate final development-only inner search froze **0.28** for the eventual full refit. This exact threshold has not been tested on the final test.

| Outer-window policy | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|
| Fixed 0,5 | 0.872 | 0.712 | 0.784 |
| Inner-selected | 0.817 | 0.782 | 0.798 |
| Current-sales rule | 0.796 | 0.813 | 0.804 |


The inner-selected threshold improved F1 over 0.50 in each outer window. The simple current-sales rule had a slightly higher mean F1 than the tuned classifier, while the selected classifier ranked high-demand articles much better by PR-AUC. This trade-off should be stated in the report; the choice of PR-AUC as primary model-selection metric remains coherent.

## Next step recorded at this checkpoint

Review this development evidence and freeze the choice. Only then set `RUN_FINAL_TEST = True` in `08_final_test_once.ipynb` and execute the final test once. Do not tune models or threshold from that result. Previous project work already inspected this publication's test, so this new protocol closes the test from the present run onward; it does not erase that history.
