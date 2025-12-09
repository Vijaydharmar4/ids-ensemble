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
   https://github.com/Vijaydharmar4/ids-ensemble/releases/tag/v1.1

## *IMPORTANT 
Use Model: cicids_multiclass.joblib. 
Create artifacts folder under project directory(ids-ensemble).
Place downloaded model inside ids-ensemble/artifacts/ before running app.

4. Run the dashboard:
   ```bash
   streamlit run app.py

