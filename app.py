# app.py
import os, io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)

# ----- Page & style -----
st.set_page_config(page_title="IDS Final Dashboard", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1300px;}
.card {background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08);
       border-radius: 14px; padding: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.12);}
.kpi {font-weight: 700; font-size: 22px; margin: 0;}
.kpi-title {opacity:.85; font-size: 12px; margin-bottom: 6px;}
.kpi-sub {opacity:.8; font-size: 12px;}
.hr {height:1px; background:rgba(255,255,255,.08); margin:8px 0 10px 0;}
h3, h4 {margin-top: 0.2rem;}
.small {font-size: 12px; opacity:.85;}
</style>
""", unsafe_allow_html=True)

# ----- helpers -----
def discover_models(folder="artifacts"):
    os.makedirs(folder, exist_ok=True)
    return [f for f in os.listdir(folder) if f.endswith(".joblib")]

def _normalize_cols(cols):
    return [str(c).replace("\t"," ").strip().lower() for c in cols]

def _clean(df: pd.DataFrame):
    """Normalize headers, keep original Label (raw_label), derive true_type and true_bin."""
    df = df.copy()
    df.columns = _normalize_cols(df.columns)

    raw_label = None
    true_type = None
    if "label" in df.columns:
        raw_label = df["label"].astype(str)
        true_type = raw_label.str.strip().str.lower()
        df = df.drop(columns=["label"])

    df = df.loc[:, ~df.columns.duplicated()]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, raw_label, true_type

def align_to_expected(df: pd.DataFrame, expected_cols):
    if not expected_cols:
        return df
    def norm(s): return str(s).replace("\t"," ").strip().lower()
    df_norm_map = {norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    matched = 0
    for exp in expected_cols:
        key = norm(exp)
        if key in df_norm_map:
            out[exp] = df[df_norm_map[key]]; matched += 1
        else:
            out[exp] = np.nan
    st.caption(f"Matched {matched}/{len(expected_cols)} expected features")
    if matched / max(1, len(expected_cols)) < 0.6:
        st.warning("Low feature match rate — check your CSV header names vs. training features.")
    return out[expected_cols]

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); return fig

def plot_roc(y_true_bin, y_scores):
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(fpr, tpr, label="ROC"); ax.plot([0,1],[0,1],"k--", linewidth=1)
    ax.set_title("ROC Curve"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout(); return fig

def plot_pr(y_true_bin, y_scores):
    prec, rec, _ = precision_recall_curve(y_true_bin, y_scores)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(rec, prec, label="PR"); ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout(); return fig

ATTACK_INFO = {
    "dos slowloris": "DoS using partial HTTP requests to exhaust server sockets.",
    "dos slowhttptest": "DoS with slow req/resp to keep many connections open.",
    "dos goldeneye": "Layer-7 HTTP DoS with rapid connections & requests.",
    "dos hulk": "Volumetric HTTP flood causing resource exhaustion.",
    "ddos": "Distributed DoS from many hosts.",
    "portscan": "Probing ports/services to enumerate open services.",
    "bot": "Compromised host contacting C2.",
    "infiltration": "Unauthorized internal access/data exfiltration.",
    "ftp-patator": "Brute-force FTP login.",
    "ssh-patator": "Brute-force SSH login.",
    "web attack xss": "Cross-Site Scripting injection.",
    "web attack sql injection": "SQL injection against backend DB.",
    "web attack brute force": "Password guessing on web login.",
    "heartbleed": "Exploit of TLS heartbeat to read server memory."
}
def _norm_attack_name(name: str) -> str:
    s = str(name).strip().lower().replace("_"," ").replace("-"," ").replace(":"," ")
    return " ".join(s.split())

def explain_attack_type(name: str) -> str:
    n = _norm_attack_name(name)
    if n in ATTACK_INFO: return ATTACK_INFO[n]
    for key, val in ATTACK_INFO.items():
        if key in n: return val
    return "Malicious traffic pattern (CICIDS)."

# ----- App -----
st.title("🛡️ IDS Using Ensemble — Multi-class (Type + Attack/Benign)")

models = discover_models("artifacts")
if not models:
    st.error("No models found in artifacts/. Train first.")
    st.stop()

with st.sidebar:
    model_choice = st.selectbox("Choose a model", models)
    load_btn = st.button("Load Model", type="primary")

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if load_btn:
    with st.spinner("🔄 Loading model..."):
        pipe = joblib.load(os.path.join("artifacts", model_choice))

        # expected cols
        expected_cols = []
        try:
            pre = pipe.named_steps.get("pre", None)
            if pre is not None and hasattr(pre, "transformers_"):
                cols_coltrans = []
                for _, _, cols in pre.transformers_:
                    if isinstance(cols, (list, tuple)): cols_coltrans += list(cols)
                expected_cols = list(dict.fromkeys(cols_coltrans))
        except Exception: pass
        if not expected_cols and hasattr(pipe, "feature_names_in_"):
            expected_cols = list(pipe.feature_names_in_)

        classes = list(getattr(pipe, "classes_", []))
        st.session_state.pipe = pipe
        st.session_state.expected_cols = expected_cols
        st.session_state.classes = classes
        st.session_state.model_loaded = True
    st.success(f"✅ Loaded model: {model_choice}")

if not st.session_state.model_loaded:
    st.info("Select and load a model to begin.")
    st.stop()

pipe = st.session_state.pipe
expected_cols = st.session_state.expected_cols
classes = st.session_state.classes

st.subheader("📤 Upload CICIDS CSV")
up = st.file_uploader("Upload CSV", type=["csv"])
if not up: st.stop()

with st.spinner("🤖 Model is processing your file..."):
    raw = pd.read_csv(up, low_memory=False)
    clean, raw_label, true_type = _clean(raw)
    X = align_to_expected(clean, expected_cols) if expected_cols else clean

    # Predict
    preds = pipe.predict(X)                          # predicted TYPE (multi-class)
    proba = None; prob_attack = None
    try:
        proba = pipe.predict_proba(X)                # [n, n_classes]
        if "benign" in classes:
            p_ben = proba[:, classes.index("benign")]
            prob_attack = 1.0 - p_ben               # derived P(attack)
    except Exception: pass

# ---------- Single row ----------
if len(X) == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Single Record Prediction")
    pred_type = str(preds[0])
    pred_attack = "ATTACK" if _norm_attack_name(pred_type) != "benign" else "BENIGN"
    st.markdown(f"**Predicted:** `{pred_attack}`  •  **Type:** `{pred_type}`")
    if prob_attack is not None:
        st.caption(f"P(attack) = {float(prob_attack[0]):.3f}")
        # top-k class probabilities
        if proba is not None:
            row = proba[0]; top_idx = np.argsort(row)[::-1][:5]
            pretty = [f"{classes[i]}: {row[i]:.3f}" for i in top_idx]
            st.caption("Top classes → " + " | ".join(pretty))
    if pred_attack == "ATTACK":
        st.info(explain_attack_type(pred_type))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Multi rows ----------
else:
    total = len(preds)
    pred_series = pd.Series(preds, dtype="object")
    # attack vs benign counts
    is_attack = pred_series.apply(lambda x: _norm_attack_name(x) != "benign")
    n_attack = int(is_attack.sum())
    n_benign = total - n_attack

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Detection Summary")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-title">Total flows</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{total:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-sub">All rows in uploaded CSV</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-title">Predicted ATTACK</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{n_attack:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">{(n_attack/total*100):.1f}%</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-title">Predicted BENIGN</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{n_benign:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-sub">{(n_benign/total*100):.1f}%</div>', unsafe_allow_html=True)
    with c4:
        if prob_attack is not None:
            st.markdown('<div class="kpi-title">Avg. P(attack)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{float(np.mean(prob_attack)):.3f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-sub">Derived from 1 - P(benign)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-title">Probabilities</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-sub">Classifier has no predict_proba</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Attack type breakdown (predicted)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Predicted Attack Types")
    atk_only = pred_series[is_attack]
    if atk_only.empty:
        st.info("No attacks predicted.")
    else:
        counts = atk_only.apply(_norm_attack_name).value_counts()
        fig, ax = plt.subplots(figsize=(4,2.8))
        counts.head(10).sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Count"); ax.set_ylabel("Attack type"); ax.set_title("Top Attack Types", fontsize=10)
        fig.tight_layout(); st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Metrics if ground truth exists (Label column)
    if true_type is not None and len(true_type) > 1:
        # make ground truth attack vs benign
        y_true_bin = true_type.apply(lambda s: 0 if _norm_attack_name(s) == "benign" else 1).values
        y_pred_bin = pred_series.apply(lambda s: 0 if _norm_attack_name(s) == "benign" else 1).values

        acc = accuracy_score(y_pred_bin, y_true_bin)
        prec = precision_score(y_true_bin, y_pred_bin, pos_label=1)
        rec  = recall_score(y_true_bin, y_pred_bin, pos_label=1)
        f1   = f1_score(y_true_bin, y_pred_bin, pos_label=1)
        roc_auc = roc_auc_score(y_true_bin, prob_attack) if prob_attack is not None else None
        pr_auc  = average_precision_score(y_true_bin, prob_attack) if prob_attack is not None else None

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Binary Metrics (Attack vs Benign)")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: st.metric("Accuracy", f"{acc:.3f}")
        with c2: st.metric("Precision", f"{prec:.3f}")
        with c3: st.metric("Recall", f"{rec:.3f}")
        with c4: st.metric("F1", f"{f1:.3f}")
        with c5: st.metric("ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")
        with c6: st.metric("PR AUC",  f"{pr_auc:.3f}"  if pr_auc  is not None else "N/A")

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).ravel()
        g1, g2, g3 = st.columns(3)
        with g1:
            st.pyplot(plot_confusion(y_true_bin, y_pred_bin, labels=[0,1]))
            st.caption(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        if prob_attack is not None:
            with g2: st.pyplot(plot_roc(y_true_bin, prob_attack))
            with g3: st.pyplot(plot_pr(y_true_bin, prob_attack))
        st.markdown('</div>', unsafe_allow_html=True)

# ----- Download predictions -----
out_df = raw.copy()
out_df["predicted_type"]  = preds
out_df["predicted_label"] = ["attack" if _norm_attack_name(p)!="benign" else "benign" for p in preds]
if 'prob_attack' in locals() and prob_attack is not None:
    out_df["attack_probability"] = prob_attack
buf = io.BytesIO(); out_df.to_csv(buf, index=False)
st.download_button("⬇️ Download predictions CSV", data=buf.getvalue(),
                   file_name="ids_predictions.csv", mime="text/csv")
st.caption("Trained as multi-class (benign + attack types). Binary metrics derive from P(attack)=1−P(benign).")
