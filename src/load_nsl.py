import pandas as pd

# 41 feature names in order (official NSL-KDD)
FEATURES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# Attack family mapping (NSL-KDD standard)
DOS = {
    "back","land","neptune","pod","smurf","teardrop",
    "apache2","udpstorm","processtable","worm",
    "mailbomb"  # present only in test; include it here
}
PROBE = {"satan","ipsweep","nmap","portsweep","mscan","saint"}
R2L = {
    "guess_passwd","ftp_write","imap","phf","multihop",
    "warezmaster","warezclient","spy","xlock","xsnoop",
    "snmpguess","snmpgetattack","httptunnel","sendmail","named"
}
U2R = {"buffer_overflow","loadmodule","rootkit","perl","ps","sqlattack","xterm"}

def _read_nsl(path: str) -> pd.DataFrame:
    # Files are comma-separated with 43 columns: 41 features + label + difficulty
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 43:
        # Fallback: whitespace-separated, just in case user has a different copy
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    if df.shape[1] != 43:
        raise ValueError(f"Expected 43 columns, found {df.shape[1]} in {path}")
    df.columns = FEATURES + ["label", "difficulty"]
    df = df.drop(columns=["difficulty"])
    # normalize labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df

def _to_family(lbl: str) -> str:
    if lbl == "normal": return "normal"
    if lbl in DOS:      return "dos"
    if lbl in PROBE:    return "probe"
    if lbl in R2L:      return "r2l"
    if lbl in U2R:      return "u2r"
    return "other"  # should be empty after mappings; helpful for spotting misses

def load_nsl(train_path="data/KDDTrain+.txt", test_path="data/KDDTest+.txt"):
    train_df = _read_nsl(train_path)
    test_df  = _read_nsl(test_path)

    # Before mapping, check unknown labels (helps catch future misses)
    KNOWN = DOS | PROBE | R2L | U2R | {"normal"}
    unk_train = train_df.loc[~train_df["label"].isin(KNOWN), "label"].value_counts()
    unk_test  = test_df.loc[~test_df["label"].isin(KNOWN), "label"].value_counts()
    if len(unk_train): print("⚠️ Unknown labels in TRAIN:", unk_train.to_dict())
    if len(unk_test):  print("⚠️ Unknown labels in TEST:",  unk_test.to_dict())

    # Map to families
    train_df["label"] = train_df["label"].map(_to_family)
    test_df["label"]  = test_df["label"].map(_to_family)

    print("✅ Loaded NSL-KDD (family labels)")
    print("Train family counts:\n", train_df["label"].value_counts())
    print("Test  family counts:\n",  test_df["label"].value_counts())
    return train_df, test_df
