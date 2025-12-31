import joblib
import os
from pathlib import Path

artifacts_dir = Path("artifacts")
files = [p.name for p in artifacts_dir.iterdir() if p.suffix in (".joblib", ".pkl")]
print(f"Files in artifacts: {files}")

for f in files:
    path = artifacts_dir / f
    print(f"\nInspecting {f}...")
    try:
        model = joblib.load(path)
        if hasattr(model, "classes_"):
            print(f"Classes: {model.classes_}")
        else:
            print("Model has no classes_ attribute.")
            
        if hasattr(model, "estimators_"):
            print(f"Ensemble estimators: {len(model.estimators_)}")
            
    except Exception as e:
        print(f"Error loading {f}: {e}")
