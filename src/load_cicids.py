# load_cicids.py
from __future__ import annotations
import os, glob
from typing import List, Tuple
import numpy as np
import pandas as pd

# Columns we should not learn on (identifiers / timestamps / dup artifacts)
DROP_CANDIDATES = {
    "flow id","flowid","flow_id",
    "src ip","source ip","srcip","source address",
    "dst ip","destination ip","dstip","destination address",
    "timestamp","time","date",
    "simillarhttp","fwd header length.1",
}

def _normalize_cols(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c = str(c).replace("\t"," ").strip().lower()
        while "  " in c:
            c = c.replace("  "," ")
        out.append(c)
    return out

def _read_one(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, low_memory=False, encoding_errors="ignore", on_bad_lines="skip")
    df.columns = _normalize_cols(df.columns)
    return df

def _find_label_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    raise ValueError("Could not find a 'Label' column in CICIDS CSV.")

def _drop_useless_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()]
    drops = [c for c in df.columns if c in DROP_CANDIDATES]
    drops += [c for c in df.columns if df[c].isna().all()]
    return df.drop(columns=list(set(drops)), errors="ignore")

def _canon_label(s: str) -> str:
    """Normalize dataset label strings into canonical attack types."""
    t = str(s).strip().lower()
    if "benign" in t: return "benign"
    if "hulk" in t: return "dos hulk"
    if "slowloris" in t: return "dos slowloris"
    if "slowhttptest" in t: return "dos slowhttptest"
    if "goldeneye" in t: return "dos goldeneye"
    if "ddos" in t: return "ddos"
    if "portscan" in t or "port scan" in t: return "portscan"
    if "heartbleed" in t: return "heartbleed"
    if "ftp" in t and "patator" in t: return "ftp-patator"
    if "ssh" in t and "patator" in t: return "ssh-patator"
    if ("brute" in t or "password" in t) and "web" in t: return "web attack brute force"
    if "sql" in t and "web" in t: return "web attack sql injection"
    if "xss" in t and "web" in t: return "web attack xss"
    if "bot" in t: return "bot"
    if "infiltration" in t: return "infiltration"
    # fallback: return cleaned original
    return t

def _read_and_clean(paths: List[str], verbose: bool = True) -> pd.DataFrame:
    frames = []
    total_rows = 0
    if verbose:
        print(f"→ Reading {len(paths)} file(s):")
        for p in paths:
            print("   •", os.path.basename(p))

    for p in paths:
        df = _read_one(p)
        lbl = _find_label_col(df)

        raw_label = df[lbl].astype(str)
        label_type = raw_label.map(_canon_label)
        label_bin = label_type.apply(lambda x: "benign" if x == "benign" else "attack")

        df = df.drop(columns=[lbl])
        df = _drop_useless_cols(df)

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        keep_cols = [c for c in df.columns if not df[c].isna().all()]
        df = df[keep_cols]

        # attach labels
        df["label_type"] = label_type.values
        df["label_binary"] = label_bin.values

        frames.append(df)
        total_rows += len(df)

    if verbose:
        print(f"   ↳ total rows after cleaning: {total_rows:,}")
    return pd.concat(frames, axis=0, ignore_index=True)

def _strat_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df): return df
    frac = min(1.0, n / len(df))
    return df.groupby("label_type", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed)
    )

def load_cicids(
    folder: str = "data/cicids",
    train_files: List[str] | None = None,
    test_files:  List[str] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    sample_train: int = 0,
    sample_test: int = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CICIDS2017 and keep multi-class 'label_type' + binary 'label_binary'.
    Returns (train_df, test_df).
    """
    if train_files or test_files:
        def norm(p: str) -> str:
            return os.path.join(folder, p) if not os.path.isabs(p) else p
        train_paths = [norm(f) for f in (train_files or [])]
        test_paths  = [norm(f) for f in (test_files  or [])]
        if not train_paths or not test_paths:
            raise ValueError("Provide BOTH train_files and test_files for time-based split.")
        train_df = _read_and_clean(train_paths, verbose=verbose)
        test_df  = _read_and_clean(test_paths,  verbose=verbose)
        if sample_train > 0: train_df = _strat_sample(train_df, sample_train, seed)
        if sample_test  > 0: test_df  = _strat_sample(test_df,  sample_test,  seed)
        if verbose:
            print("✅ Time split loaded")
            print("Train label_type counts:\n", train_df["label_type"].value_counts())
            print("Test  label_type counts:\n", test_df["label_type"].value_counts())
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs found under {folder}. Put CICIDS CSVs there.")
    if verbose:
        print(f"→ Discovering CSVs under {folder} ...")
        for p in paths: print("   •", os.path.basename(p))

    df = _read_and_clean(paths, verbose=verbose)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label_type"]
    )
    if sample_train > 0: train_df = _strat_sample(train_df, sample_train, seed)
    if sample_test  > 0: test_df  = _strat_sample(test_df,  sample_test,  seed)

    if verbose:
        print(f"✅ Random split: {len(train_df):,} train, {len(test_df):,} test")
        print("Train label_type counts:\n", train_df["label_type"].value_counts())
        print("Test  label_type counts:\n",  test_df["label_type"].value_counts())
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
