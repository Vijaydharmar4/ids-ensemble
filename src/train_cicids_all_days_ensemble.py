import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from load_cicids import load_cicids  # random 80/20 split when no file lists are passed

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def strat_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n <= 0 or n >= len(df): return df
    frac = min(1.0, n / len(df))
    return df.groupby("label", group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=seed))


def build_preproc(num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols)
    ], remainder="drop")


def build_ensemble(n_estimators_rf: int, seed: int, weights: tuple[int, int, int]) -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators_rf,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed
    )
    gb = GradientBoostingClassifier(random_state=seed)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=list(weights)
    )


def main():
    ap = argparse.ArgumentParser(description="ALL-days CICIDS2017 soft-voting ensemble (random 80/20 split).")
    ap.add_argument("--data_dir", default="data/cicids", help="Folder with CICIDS CSVs")
    ap.add_argument("--n_estimators_rf", type=int, default=300, help="Trees for RandomForest (ensemble member)")
    ap.add_argument("--weights", default="2,1,1", help="Voting weights for RF,GB,LR (e.g., 2,1,1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_train", type=int, default=0, help="Stratified sample size for train (0=all)")
    ap.add_argument("--sample_test", type=int, default=0, help="Stratified sample size for test (0=all)")
    ap.add_argument("--outprefix", default="cicids_all_days_ensemble", help="Filename prefix for saved artifacts")
    args = ap.parse_args()

    weights = tuple(int(x) for x in args.weights.split(","))
    assert len(weights) == 3, "Provide three weights like 2,1,1"

    # Load all CSVs (random 80/20 split)
    train_df, test_df = load_cicids(args.data_dir, sample_train=args.sample_train, sample_test=args.sample_test)

    y_train, X_train = train_df["label"], train_df.drop(columns=["label"])
    y_test,  X_test  = test_df["label"],  test_df.drop(columns=["label"])

    num_cols = list(X_train.columns)
    pre = build_preproc(num_cols)
    ens = build_ensemble(args.n_estimators_rf, args.seed, weights)

    pipe = Pipeline([("pre", pre), ("ens", ens)])
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Probabilities for AUCs
    y_scores = None
    try:
        proba = pipe.predict_proba(X_test)
        pos_idx = list(pipe.classes_).index("attack")
        y_scores = proba[:, pos_idx]
    except Exception:
        pass

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    metrics = {
        "task": "cicids-all-days-ensemble",
        "seed": args.seed,
        "n_estimators_rf": args.n_estimators_rf,
        "weights_rf_gb_lr": list(weights),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_label_counts": y_train.value_counts().to_dict(),
        "test_label_counts":  y_test.value_counts().to_dict(),
        "accuracy": float(acc),
        "precision_macro": float(pr),
        "recall_macro": float(rc),
        "f1_macro": float(f1),
        "per_class_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "confusion_matrix": {
            "labels_in_order": list(pipe.classes_),
            "matrix": confusion_matrix(y_test, y_pred, labels=list(pipe.classes_)).tolist()
        }
    }
    if y_scores is not None:
        y_bin = (y_test.values == "attack").astype(int)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_bin, y_scores))
            metrics["pr_auc"]  = float(average_precision_score(y_bin, y_scores))
        except Exception:
            pass

    outdir = Path("artifacts"); outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"{args.outprefix}.joblib"
    metrics_path = outdir / f"{args.outprefix}_metrics.json"

    joblib.dump(pipe, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Trained ALL-days CICIDS (Ensemble RF+GB+LR)")
    print(f"Saved model   → {model_path}")
    print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
