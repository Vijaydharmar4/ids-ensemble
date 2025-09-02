# 🛡️ Intrusion Detection System (IDS) using Ensemble Learning

An Intrusion Detection System built with **ensemble learning** on the **CICIDS2017 dataset**, achieving ~99% accuracy.  
Supports both **binary classification** (Attack vs Benign) and **multi-class classification** (e.g., DoS Hulk, DDoS, PortScan, Bot, SQL Injection, etc.).

---

## 🚀 Features
- **Ensemble Model** → Random Forest + Gradient Boosting + Logistic Regression (soft voting).
- **Multi-class Attack Detection** → Identifies attack types (DoS, DDoS, PortScan, Bot, etc.).
- **Streamlit Dashboard** → Real-time predictions, metrics, and visualizations.
- **Metrics & Visuals**:
  - Accuracy, Precision, Recall, F1, ROC AUC, PR AUC
  - Confusion Matrix, ROC Curve, PR Curve
  - Attack-type breakdown and explanations
- **Demo-ready CSVs** → Supports single-row and small test datasets.

---

## 📂 Project Structure

ids-ensemble/
├── src/                   # Source code
│   ├── app.py             # Streamlit dashboard
│   ├── load_cicids.py     # Data loading & preprocessing
│   └── train_ids_multiclass.py   # Training script
├── artifacts/             # Trained models (.joblib) [not pushed to GitHub]
├── data/                  # Small demo CSVs (not full dataset)
├── requirements.txt       # Dependencies
└── README.md              # Documentation


---

## ⚡ Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Vijaydharmar4/ids-ensemble.git
   cd ids-ensemble

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. ## 📥 Download Trained Models
   ```bash
   https://github.com/Vijaydharmar4/ids-ensemble/releases/tag/v1.0
Place downloaded model inside ids-ensemble/artifacts/ before running app.

4. Run the dashboard:
   ```bash
   streamlit run src/app.py

