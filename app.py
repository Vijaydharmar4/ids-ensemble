# app.py — IDS Dashboard (Auto-load model, no emoji, charts + robust preprocessing)

import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# ----------------- Page config & CSS -----------------
st.set_page_config(page_title="IDS Ensemble Dashboard", layout="wide")

st.markdown("""
<style>
body { background-color: #0f1419 !important; color: #e8edf2 !important; }
.block-container { padding-top: 1rem !important; max-width: 1300px; }
.big-header {
    background: linear-gradient(90deg, #152030, #0e1622);
    padding: 24px;
    border-radius: 12px;
    border: 1px solid rgba(100,200,255,.08);
    margin-bottom: 18px;
}
.info-box { background:#151d22; padding:14px; border-radius:10px; border:1px solid rgba(255,255,255,.04); margin-bottom:16px; color:#c6d6e5; }
.card { background:#1c2633; padding:18px; border-radius:12px; border:1px solid rgba(255,255,255,.04); margin-bottom:14px; }
.kpi { font-weight:700; font-size:20px; color:#4a9eff; }
.small { font-size:13px; color:#9eaebc; }
</style>
""", unsafe_allow_html=True)

# ----------------- Utility helpers -----------------
def _normalize_cols(cols):
    return [str(c).replace("\t", " ").strip().lower() for c in cols]

def _norm_attack_name(name: str) -> str:
    s = str(name).strip().lower().replace("_"," ").replace("-"," ").replace(":"," ")
    return " ".join(s.split())

def align_features_auto(df: pd.DataFrame, expected_cols):
    """
    Align DataFrame to expected columns:
    - normalize incoming column names (lowercase, trim)
    - if an expected column missing -> create and fill with 0
    - drop extras, keep expected column order
    """
    df = df.copy()
    # normalize source columns
    src_norm_map = {str(c).strip().lower().replace("\t"," "): c for c in df.columns}
    expected_norm = [str(c).strip().lower().replace("\t"," ") for c in expected_cols]

    aligned = pd.DataFrame(index=df.index)
    missing = []
    for exp_col, exp_norm in zip(expected_cols, expected_norm):
        if exp_norm in src_norm_map:
            aligned[exp_col] = df[src_norm_map[exp_norm]]
        else:
            aligned[exp_col] = 0
            missing.append(exp_col)

    if missing:
        st.warning(f"Missing {len(missing)} expected columns were auto-created (filled with 0).")
    return aligned

def clean_numeric(df: pd.DataFrame):
    """
    Convert non-numeric to NaN, replace infinities, clip extremes.
    Returns cleaned dataframe.
    """
    df = df.copy()
    # convert everything to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')
    # replace ±inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # clip extremes to safe float64 range
    df = df.clip(lower=-1e308, upper=1e308)
    return df

# ----------------- Info text (no emoji) -----------------
st.markdown("""
<div class="big-header">
    <h1>Intrusion Detection System — Ensemble Multi-Class</h1>
    <p class="small">This dashboard uses a pre-trained ensemble model (cicids_multiclass) to label network flows
    as benign or specific attack types. Upload a CICIDS-format CSV (flow-level features) to view predictions and charts.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>About:</strong> The CICIDS2017-style datasets contain flow-based network features (derived with CICFlowMeter).
Typical features include packet/byte counts, timing (inter-arrival) statistics, flags counts, window sizes, and derived ratios.
This dashboard auto-aligns uploaded CSV columns to the model's expected features, fills missing features with zeros,
and cleans numeric issues before inference.
</div>
""", unsafe_allow_html=True)

# ----------------- Auto-load model (no selection) -----------------
MODEL_PATH = "artifacts/cicids_multiclass.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Model not found: artifacts/cicids_multiclass.joblib. Place the model in the artifacts/ folder.")
    st.stop()

with st.spinner("Loading model..."):
    pipe = joblib.load(MODEL_PATH)

# Attempt to infer expected columns: prefer feature_names_in_
if hasattr(pipe, "feature_names_in_"):
    expected_cols = list(pipe.feature_names_in_)
else:
    # try to extract from a column transformer (if present)
    expected_cols = []
    try:
        pre = pipe.named_steps.get("pre", None)
        if pre is not None and hasattr(pre, "transformers_"):
            cols_coltrans = []
            for _, _, cols in pre.transformers_:
                if isinstance(cols, (list, tuple)):
                    cols_coltrans += list(cols)
            expected_cols = list(dict.fromkeys(cols_coltrans))
    except Exception:
        expected_cols = []

if not expected_cols:
    st.warning("Warning: model has no explicit feature list. Predictions may fail if the CSV columns do not match training features.")

# ----------------- File uploader -----------------
st.subheader("Upload dataset (CSV)")
uploaded = st.file_uploader("Choose a CICIDS-style CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

with st.spinner("Reading file..."):
    try:
        raw = pd.read_csv(uploaded, low_memory=False)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# keep a copy of raw for download & reference
raw_for_download = raw.copy()

# normalize column names for display and processing
raw.columns = _normalize_cols(raw.columns)

# ----------------- Align features & clean -----------------
if expected_cols:
    X = align_features_auto(raw, expected_cols)
else:
    # if model has no expected cols, try to use the raw dataframe as-is (but normalize columns)
    X = raw.copy()

# Convert to numeric, replace infinities, clip extremely large values
X = clean_numeric(X)

# ----------------- Predict (safe) -----------------
with st.spinner("Running model predictions..."):
    try:
        preds = pipe.predict(X)
    except Exception as e:
        st.error("Prediction failed. Attempting a safer numeric conversion and retry.")
        # extra safety: coerce to numeric again
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(0)
        try:
            preds = pipe.predict(X)
        except Exception as e2:
            st.error(f"Prediction still failed: {e2}")
            st.stop()

    # try predict_proba (if available)
    proba = None
    prob_attack = None
    try:
        proba = pipe.predict_proba(X)
        # try find index for benign label if present
        classes = list(getattr(pipe, "classes_", []))
        if "benign" in classes:
            p_ben = proba[:, classes.index("benign")]
            prob_attack = 1.0 - p_ben
    except Exception:
        proba = None
        prob_attack = None

# ----------------- Build annotated dataframe and KPIs -----------------
pred_series = pd.Series(preds).astype(str)
is_attack = pred_series.apply(lambda x: _norm_attack_name(x) != "benign")
total = len(pred_series)
n_attack = int(is_attack.sum())
n_benign = total - n_attack

# KPIs
st.markdown("<div class='card'><h3>Overview</h3></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='card'><div class='kpi'>%s</div><div class='small'>Total records</div></div>" % f"{total:,}", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><div class='kpi'>%s</div><div class='small'>Predicted attacks</div></div>" % f"{n_attack:,}", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><div class='kpi'>%s</div><div class='small'>Predicted benign</div></div>" % f"{n_benign:,}", unsafe_allow_html=True)

# ----------------- Charts -----------------
st.markdown("<div class='card'><h3>Attack vs Benign (pie)</h3>", unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(4,3), facecolor="#0f1419")
sizes = [n_attack, n_benign]
labels = ["Attack", "Benign"]
colors = ["#ff6b6b", "#4a9eff"]
ax1.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(p * sum(sizes) / 100):,})", colors=colors, textprops={'color':'white'})
ax1.set_aspect('equal')
plt.tight_layout()
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

# Attack types bar chart
st.markdown("<div class='card'><h3>Top Predicted Attack Types</h3>", unsafe_allow_html=True)
atk_only = pred_series[is_attack]
if atk_only.empty:
    st.info("No attacks predicted in the uploaded data.")
else:
    counts = atk_only.apply(_norm_attack_name).value_counts()
    fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.4 * min(10, len(counts)))), facecolor="#0f1419")
    sns.barplot(x=counts.values[:15], y=counts.index[:15], palette="coolwarm", ax=ax2)
    ax2.set_xlabel("Count", color="white")
    ax2.set_ylabel("Attack Type", color="white")
    ax2.tick_params(colors="white")
    st.pyplot(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# Additional charts based on available columns
st.markdown("<div class='card'><h3>Additional Distributions</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

# Histogram of packet length mean (if exists)
with col1:
    candidates = [c for c in X.columns if "packet" in c and ("mean" in c or "average" in c)]
    if candidates:
        col = candidates[0]
        fig3, ax3 = plt.subplots(figsize=(5,3), facecolor="#0f1419")
        sns.histplot(X[col].dropna(), bins=50, ax=ax3, color="#4a9eff")
        ax3.set_title(f"Distribution of {col}", color="white")
        ax3.tick_params(colors="white")
        st.pyplot(fig3)
    else:
        st.info("No packet-length mean column found for histogram.")

# Flow duration histogram (if exists)
with col2:
    duration_cols = [c for c in X.columns if "flow duration" in c or "duration" in c]
    if duration_cols:
        col = duration_cols[0]
        fig4, ax4 = plt.subplots(figsize=(5,3), facecolor="#0f1419")
        sns.histplot(X[col].dropna(), bins=50, ax=ax4, color="#ffb86b")
        ax4.set_title(f"Distribution of {col}", color="white")
        ax4.tick_params(colors="white")
        st.pyplot(fig4)
    else:
        st.info("No flow duration column found for histogram.")
st.markdown("</div>", unsafe_allow_html=True)

# If protocol-like column exists, show protocol distribution
proto_candidates = [c for c in raw.columns if any(k in c for k in ["protocol", "proto", "ip_proto", "protocol name"])]
if proto_candidates:
    proto_col = proto_candidates[0]
    st.markdown("<div class='card'><h3>Protocol Distribution</h3>", unsafe_allow_html=True)
    proto_counts = raw[proto_col].astype(str).value_counts().head(15)
    fig5, ax5 = plt.subplots(figsize=(6,3), facecolor="#0f1419")
    sns.barplot(x=proto_counts.values, y=proto_counts.index, ax=ax5, palette="magma")
    ax5.tick_params(colors="white")
    st.pyplot(fig5)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Ground truth metrics (if label column present) -----------------
label_present = False
if "label" in raw.columns:
    label_present = True
    true_raw = raw["label"].astype(str).str.strip().str.lower()
    # create binary arrays
    y_true_bin = np.array([0 if _norm_attack_name(s) == "benign" else 1 for s in true_raw])
    y_pred_bin = np.array([0 if _norm_attack_name(s) == "benign" else 1 for s in pred_series])

    st.markdown("<div class='card'><h3>Binary Metrics (if ground truth present)</h3>", unsafe_allow_html=True)
    try:
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cmat = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{acc:.3f}")
        cols[1].metric("Precision", f"{prec:.3f}")
        cols[2].metric("Recall", f"{rec:.3f}")
        cols[3].metric("F1", f"{f1:.3f}")

        # confusion matrix
        figc, axc = plt.subplots(figsize=(3,3), facecolor="#0f1419")
        sns.heatmap(cmat, annot=True, fmt="d", cmap="Blues", ax=axc, cbar=False)
        axc.set_xlabel("Predicted"); axc.set_ylabel("Actual")
        st.pyplot(figc)
    except Exception as e:
        st.warning(f"Could not compute metrics: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Prepare annotated CSV for download -----------------
annotated = raw_for_download.copy()
annotated["predicted_type"] = pred_series.values
annotated["predicted_label"] = ["attack" if _norm_attack_name(p) != "benign" else "benign" for p in pred_series]
if prob_attack is not None:
    annotated["attack_probability"] = prob_attack

buf = io.BytesIO()
annotated.to_csv(buf, index=False)
st.download_button("Download predictions CSV", data=buf.getvalue(), file_name="ids_predictions.csv", mime="text/csv")

# ----------------- Footer note -----------------
st.caption("Model: cicids_multiclass (preloaded). Uploaded data was auto-aligned to model features, missing features were filled with zeros, and numeric data was cleaned before inference. Only use this tool on data/networks you are authorized to analyze.")
