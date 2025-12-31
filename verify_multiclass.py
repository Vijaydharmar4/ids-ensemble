import joblib
import pandas as pd
import numpy as np
import os

model_path = "artifacts/cicids_multiclass.joblib"
data_path = "Test_Input_Data/sample_input.csv"

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

print(f"Loading {model_path}...")
model = joblib.load(model_path)
print(f"Classes: {model.classes_}")

if not os.path.exists(data_path):
    print(f"Data not found: {data_path}")
    exit(1)

print(f"Loading {data_path}...")
df = pd.read_csv(data_path)

# Mock cleaning/alignment as per backend
def align_features(df, model):
    if hasattr(model, "feature_names_in_"):
        expected = model.feature_names_in_
        # simple alignment
        aligned = pd.DataFrame(0, index=df.index, columns=expected)
        # map common columns
        for c in df.columns:
            # normalize
            norm_c = c.strip().lower().replace("\t", " ")
            for exp in expected:
                if exp.strip().lower() == norm_c:
                    aligned[exp] = df[c]
        return aligned
    return df

X = align_features(df, model)
X = X.fillna(0)

print("Predicting...")
preds = model.predict(X)
unique_preds = np.unique(preds)
print(f"Unique predictions: {unique_preds}")
