import React from 'react';
import './BinaryMetrics.css';

export function BinaryMetrics({ metrics, confusion }) {
  return (
    <div className="card">
      <h3>📈 Binary Metrics (Attack vs Benign)</h3>
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">Accuracy</span>
          <span className="metric-value">{metrics.accuracy.toFixed(3)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Precision</span>
          <span className="metric-value">{metrics.precision.toFixed(3)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Recall</span>
          <span className="metric-value">{metrics.recall.toFixed(3)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">F1</span>
          <span className="metric-value">{metrics.f1.toFixed(3)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">ROC AUC</span>
          <span className="metric-value">{metrics.roc_auc?.toFixed(3) || 'N/A'}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">PR AUC</span>
          <span className="metric-value">{metrics.pr_auc?.toFixed(3) || 'N/A'}</span>
        </div>
      </div>
      {confusion && (
        <div className="confusion-matrix">
          <h4>Confusion Matrix</h4>
          <table className="matrix-table">
            <tbody>
              <tr>
                <td>TN: {confusion.tn}</td>
                <td>FP: {confusion.fp}</td>
              </tr>
              <tr>
                <td>FN: {confusion.fn}</td>
                <td>TP: {confusion.tp}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}