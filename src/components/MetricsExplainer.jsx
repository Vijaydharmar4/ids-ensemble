import React, { useState } from 'react';
import './MetricsExplainer.css';

export function MetricsExplainer() {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="expander">
      <button onClick={() => setExpanded(!expanded)} className="expander-btn">
        📘 Explanation of Binary Metrics (Attack vs Benign)
      </button>
      {expanded && (
        <div className="expander-content">
          <h4>1️⃣ Accuracy</h4>
          <p>Accuracy = (TP + TN) / Total samples. Indicates overall percentage of correct predictions.</p>
          
          <h4>2️⃣ Precision</h4>
          <p>Precision = TP / (TP + FP). Out of everything predicted as attack, how many were truly attacks?</p>
          
          <h4>3️⃣ Recall</h4>
          <p>Recall = TP / (TP + FN). Out of all actual attacks, how many were detected correctly?</p>
          
          <h4>4️⃣ F1 Score</h4>
          <p>F1 = 2 × (Precision × Recall) / (Precision + Recall). Balanced measure.</p>
          
          <h4>5️⃣ ROC AUC</h4>
          <p>Evaluates model performance across every classification threshold. Ranges from 0.5 (random) to 1.0 (perfect).</p>
          
          <h4>6️⃣ PR AUC</h4>
          <p>More informative than ROC AUC for imbalanced datasets. Shows trade-off between precision and recall.</p>
        </div>
      )}
    </div>
  );
}