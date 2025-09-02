import json
from pathlib import Path
import numpy as np
import pandas as pd
from load_nsl import load_nsl   # loads with family labels (normal/dos/probe/r2l/u2r)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import joblib

def main():
    outdir = Path("artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- load NSL-KDD (5-class family labels) ----------
    train_df, test_df = load_nsl("data/KDDTrain+.txt", "data/KDDTest+.txt")
    y_train, X_train = train_df['label'], train_df.drop(columns=['label'])
    y_test,  X_test  = test_df['label'],  test_df.drop(columns=['label'])

    # ---------- preprocessing ----------
    CAT_COLS = ["protocol_type", "service", "flag"]
    NUM_COLS = [c for c in X_train.columns if c not in CAT_COLS]

    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    pre = ColumnTransformer([
        ('num', numeric_pipe, NUM_COLS),
        ('cat', cat_pipe, CAT_COLS)
    ], remainder='drop')

    # ---------- class weights (handle imbalance across 5 classes) ----------
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw_dict = {cls: w for cls, w in zip(classes, weights)}

    # ---------- model ----------
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight=cw_dict
    )

    pipe = Pipeline([('pre', pre), ('rf', clf)])

    # ---------- train ----------
    pipe.fit(X_train, y_train)

    # ---------- test ----------
    y_pred = pipe.predict(X_test)

    # ---------- metrics (multiclass) ----------
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    report = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(classes)).tolist()

    metrics = {
        "task": "nsl-kdd-family-5class",
        "classes": list(classes),
        "accuracy": acc,
        "precision_macro": pr,
        "recall_macro": rc,
        "f1_macro": f1,
        "per_class_report": report,
        "confusion_matrix": {
            "labels_in_order": list(classes),
            "matrix": cm
        }
    }

    # ---------- save ----------
    joblib.dump(pipe, outdir / 'nsl_family_rf.joblib')
    with open(outdir / 'nsl_family_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("✅ Trained NSL-KDD (family 5-class) RF")
    print(f"Saved model → {outdir / 'nsl_family_rf.joblib'}")
    print(f"Saved metrics → {outdir / 'nsl_family_metrics.json'}")

if __name__ == '__main__':
    main()
