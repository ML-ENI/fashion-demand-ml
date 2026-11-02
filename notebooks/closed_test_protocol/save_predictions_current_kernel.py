"""Run with exec(...) inside the already-executed final-test notebook kernel.

This exports the predictions already held in memory. It does not fit a model
or recompute a test score.
"""

from hashlib import sha256
import json
import numpy as np
import pandas as pd

needed = [
    "output_dir", "full_test", "regression_prediction",
    "classification_probability", "current_sales_probability",
    "run_dir", "HIGH_DEMAND_THRESHOLD", "TARGET_REGRESSION", "TARGET_CLASS",
]
missing = [name for name in needed if name not in globals()]
if missing:
    raise RuntimeError(
        "Run this inside the same final-test notebook kernel. "
        f"Missing variables: {missing}"
    )

metrics_path = output_dir / "final_test_metrics.json"
predictions_path = output_dir / "final_test_predictions.csv.gz"
if not metrics_path.is_file():
    raise FileNotFoundError("Final-test metrics have not been saved.")
if predictions_path.exists():
    raise FileExistsError("Predictions were already exported.")
record = json.loads(metrics_path.read_text(encoding="utf-8"))
if record["publication_run"] != run_dir.name or record["test_rows"] != len(full_test):
    raise ValueError("The live kernel and saved test record do not match.")
if len(regression_prediction) != len(full_test) or len(classification_probability) != len(full_test):
    raise ValueError("Live predictions and test rows have different lengths.")

table = pd.DataFrame({
    "target_week": full_test["target_week"].dt.strftime("%Y-%m-%d").to_numpy(),
    "actual_next_week_sales": full_test[TARGET_REGRESSION].to_numpy(),
    "predicted_next_week_sales": np.maximum(regression_prediction, 0),
    "persistence_prediction": full_test["sales"].to_numpy(),
    "actual_high_demand": full_test[TARGET_CLASS].to_numpy(dtype="int8"),
    "predicted_high_demand_probability": classification_probability,
    "current_sales_rule": current_sales_probability,
})
table.to_csv(predictions_path, index=False, compression="gzip")
digest = sha256()
with predictions_path.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
record["predictions_file"] = predictions_path.name
record["predictions_sha256"] = digest.hexdigest()
record["high_demand_units_threshold"] = int(HIGH_DEMAND_THRESHOLD)
metrics_path.write_text(
    json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)
print("Saved existing final-test predictions:", predictions_path)
print("No model was retrained.")
