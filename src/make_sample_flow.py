import argparse, os
import numpy as np
import pandas as pd
import joblib

DROP_CANDIDATES = {
    "flow id","flowid","flow_id",
    "src ip","source ip","srcip","source address",
    "dst ip","destination ip","dstip","destination address",
    "timestamp","time","date",
    "simillarhttp","fwd header length.1",
}

def _normalize_cols(cols):
    out = []
    for c in cols:
        c = str(c).replace("\t"," ").strip().lower()
        while "  " in c:
            c = c.replace("  "," ")
        out.append(c)
    return out

def _clean(df):
    df = df.copy()
    df.columns = _normalize_cols(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    drops = [c for c in df.columns if c in DROP_CANDIDATES]
    drops += [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=list(set(drops)), errors="ignore")
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def main():
    ap = argparse.ArgumentParser(description="Create a small sample_flow.csv from a CICIDS CSV.")
    ap.add_argument("--from_file", required=True, help="Path to a CICIDS CSV")
    ap.add_argument("--model", default="artifacts/cicids_all_days_ensemble.joblib", help="Trained model to get expected feature list")
    ap.add_argument("--nrows", type=int, default=1000, help="How many top rows to read")
    ap.add_argument("--out", default="sample_flow.csv", help="Output CSV path")
    args = ap.parse_args()

    if not os.path.exists(args.from_file):
        raise SystemExit(f"File not found: {args.from_file}")

    # get expected features from model
    pipe = joblib.load(args.model)
    expected = list(pipe.named_steps["pre"].transformers_[0][2])

    df = pd.read_csv(args.from_file, low_memory=False, nrows=args.nrows)
    df = _clean(df)

    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    df = df[expected]

    df.to_csv(args.out, index=False)
    print(f"✅ Wrote {len(df)} rows → {args.out}")
    print(f"(Columns aligned to model: {len(expected)})")

if __name__ == "__main__":
    main()
