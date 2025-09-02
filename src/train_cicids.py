# time-based CICIDS2017 training:
# Train on Monday + Tuesday, Test on Wednesday

import os, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

from load_cicids import load_cicids

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import joblib


def _pick_day_files(folder: str, day_prefix: str) -> list[str]:
    """
    Find CSVs for a given day (prefix like 'Monday', 'Tuesday', 'Wednesday').
    Handles small case/name differences by globbing.
    Returns a list of basenames (relative to folder).
    """
    # Prefer official naming that ends with ISCX.csv
    patterns = [
        f"{day_prefix}*ISCX.csv",
        f"{day_prefix}*.csv",
        f"{day_prefix.lower()}*ISCX.csv",
        f"{day_prefix.lower()}*.csv",
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(folder, pat)))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"No CSVs found for {day_prefix} under {folder}. "
            f"Expected something like '{day_prefix}-WorkingHours.pcap_ISCX.csv'."
        )
    # Return basenames because load_cicids joins with folder
    return [os.path.basename(p) for p in matches]


def main():
    data_dir = "data/cicids"

    # --- Pick files for time-based split
    monday_files    = _pick_day_files(data_dir, "Monday")
    tuesday_files   = _pick_day_files(data_dir, "Tuesday")
    wednesday_files = _pick_day_files(data_dir, "Wednesday")

    # Use all Monday+Tuesday for training, all Wednesday for testing
    train_files = monday_files + tuesday_files
    test_files  = wednesday_files

    # --- Load (time-based split)
    train_df, test_df = load_cicids(
        data_dir,
        train_files=train_files,
        test_files=test_files
    )

    y_train, X_train = train_df["label"], train_df.drop(columns=["label"])
    y_test,  X_test  = test_df["label"],  test_df.drop(columns=["label"])

    # --- Preprocessing: numeric-only (loader already removed strings)
    NUM_COLS = list(X_train.columns)
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), NUM_COLS)
    ], remainder="drop")

    # --- Class weights (handle imbalance)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw_dict = {cls: w for cls, w in zip(classes, weights)}

    # --- Model (tweak n_estimators if training feels slow)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight=cw_dict
    )
    pipe = Pipeline([("pre", pre), ("rf", clf)])

    # --- Train
    pipe.fit(X_train, y_train)

    # --- Predict
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    pos_idx = list(pipe.classes_).index("attack")
    y_scores = proba[:, pos_idx]
    y_bin = (y_test.values == "attack").astype(int)

    # --- Metrics
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "task": "cicids2017-binary-rf-timesplit-mon+tue->wed",
        "train_files": train_files,
        "test_files": test_files,
        "train_label_counts": y_train.value_counts().to_dict(),
        "test_label_counts":  y_test.value_counts().to_dict(),
        "accuracy": acc,
        "precision_macro": pr,
        "recall_macro": rc,
        "f1_macro": f1,
        "per_class_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "confusion_matrix": {
            "labels_in_order": list(classes),
            "matrix": confusion_matrix(y_test, y_pred, labels=list(classes)).tolist()
        }
    }
    # Ranking metrics (useful with imbalance)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_bin, y_scores))
        metrics["pr_auc"]  = float(average_precision_score(y_bin, y_scores))
    except Exception:
        pass

    # --- Save
    outdir = Path("artifacts"); outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "cicids_timesplit_rf.joblib")
    with open(outdir / "cicids_timesplit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Trained CICIDS2017 (Mon+Tue → Wed) RF")
    print(f"Saved model → {outdir / 'cicids_timesplit_rf.joblib'}")
    print(f"Saved metrics → {outdir / 'cicids_timesplit_metrics.json'}")


if __name__ == "__main__":
    main()
