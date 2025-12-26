import React from 'react';
import './DetectionSummary.css';

export function DetectionSummary({ stats }) {
  return (
    <div className="card">
      <h3>📊 Detection Summary</h3>
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-title">Total flows</div>
          <div className="kpi-value">{stats.total.toLocaleString()}</div>
          <div className="kpi-sub">All rows in uploaded CSV</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-title">Predicted ATTACK</div>
          <div className="kpi-value">{stats.n_attack.toLocaleString()}</div>
          <div className="kpi-sub">{((stats.n_attack / stats.total) * 100).toFixed(1)}%</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-title">Predicted BENIGN</div>
          <div className="kpi-value">{stats.n_benign.toLocaleString()}</div>
          <div className="kpi-sub">{((stats.n_benign / stats.total) * 100).toFixed(1)}%</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-title">Avg. P(attack)</div>
          <div className="kpi-value">{stats.avg_prob_attack?.toFixed(3) || 'N/A'}</div>
          <div className="kpi-sub">Derived from 1 - P(benign)</div>
        </div>
      </div>
    </div>
  );
}