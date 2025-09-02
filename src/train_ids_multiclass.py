# train_ids_multiclass.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from load_cicids import load_cicids

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)

def build_preproc(num_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols)
    ], remainder="drop")

def build_ensemble(n_estimators_rf: int, seed: int, weights):
    rf = RandomForestClassifier(
        n_estimators=n_estimators_rf,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )
    gb = GradientBoostingClassifier(random_state=seed)  # no class_weight
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed, solver="lbfgs",
        multi_class="auto"
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=list(weights)
    )

def main():
    ap = argparse.ArgumentParser(description="Train multi-class CICIDS ensemble (RF+GB+LR).")
    ap.add_argument("--data_dir", default="data/cicids")
    ap.add_argument("--n_estimators_rf", type=int, default=400)
    ap.add_argument("--weights", default="2,1,1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_train", type=int, default=0)
    ap.add_argument("--sample_test", type=int, default=0)
    ap.add_argument("--outprefix", default="cicids_multiclass")
    args = ap.parse_args()

    weights = tuple(int(x) for x in args.weights.split(","))

    train_df, test_df = load_cicids(
        folder=args.data_dir,
        sample_train=args.sample_train,
        sample_test=args.sample_test,
        seed=args.seed,
        verbose=True
    )

    y_train = train_df["label_type"]
    y_test  = test_df["label_type"]
    X_train = train_df.drop(columns=["label_type","label_binary"])
    X_test  = test_df.drop(columns=["label_type","label_binary"])

    num_cols = list(X_train.columns)
    pipe = Pipeline([("pre", build_preproc(num_cols)),
                     ("ens", build_ensemble(args.n_estimators_rf, args.seed, weights))])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # For binary metrics/curves: use P(attack) = 1 - P(benign)
    proba = None
    prob_attack = None
    try:
        proba = pipe.predict_proba(X_test)  # shape [n, n_classes]
        classes = list(pipe.classes_)
        if "benign" in classes:
            p_ben = proba[:, classes.index("benign")]
            prob_attack = 1.0 - p_ben
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classes": list(pipe.classes_),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels_in_order": list(pipe.classes_),
            "matrix": confusion_matrix(y_test, y_pred, labels=list(pipe.classes_)).tolist()
        }
    }
    if prob_attack is not None:
        y_bin_true = (y_test.values != "benign").astype(int)
        try:
            metrics["roc_auc_attack_vs_benign"] = float(roc_auc_score(y_bin_true, prob_attack))
            metrics["pr_auc_attack_vs_benign"]  = float(average_precision_score(y_bin_true, prob_attack))
        except Exception:
            pass

    outdir = Path("artifacts"); outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"{args.outprefix}.joblib"
    metrics_path = outdir / f"{args.outprefix}_metrics.json"

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Trained multi-class CICIDS ensemble")
    print(f"Saved model   → {model_path}")
    print(f"Saved metrics → {metrics_path}")

if __name__ == "__main__":
    main()
