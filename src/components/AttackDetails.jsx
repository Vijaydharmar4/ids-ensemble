import React, { useState } from 'react';
import { getAttackInfo } from '../utils/attackInfo';
import './AttackDetails.css';

export function AttackDetails({ attackCounts }) {
  const [expandedAttacks, setExpandedAttacks] = useState(new Set());

  if (!attackCounts || (Array.isArray(attackCounts) && attackCounts.length === 0)) {
    return (
      <div className="card attack-details-container">
        <h3>🛡️ Attack Analysis</h3>
        <div className="no-attacks">
          <div className="success-icon">✅</div>
          <h4>No Attacks Detected</h4>
          <p>Your network traffic appears to be clean. All flows were classified as benign.</p>
        </div>
      </div>
    );
  }

  // Convert to array if object
  const attacksArray = Array.isArray(attackCounts)
    ? attackCounts
    : Object.entries(attackCounts).map(([name, count]) => ({ name, count }));

  // Sort by count (descending)
  const sortedAttacks = attacksArray
    .filter(attack => attack.name && attack.name.toLowerCase() !== 'benign')
    .sort((a, b) => (b.count || 0) - (a.count || 0));

  const totalAttacks = sortedAttacks.reduce((sum, attack) => sum + (attack.count || 0), 0);

  const toggleExpand = (attackName) => {
    const newExpanded = new Set(expandedAttacks);
    if (newExpanded.has(attackName)) {
      newExpanded.delete(attackName);
    } else {
      newExpanded.add(attackName);
    }
    setExpandedAttacks(newExpanded);
  };

  return (
    <div className="card attack-details-container">
      <div className="attack-details-header">
        <h3>
          <span style={{ fontSize: '2rem', marginRight: '10px' }}>🛡️</span>
          Detected Attacks Analysis
        </h3>
        <div className="attack-summary-badge">
          <span className="badge-icon">⚠️</span>
          <span>{sortedAttacks.length} Attack Type{sortedAttacks.length !== 1 ? 's' : ''}</span>
          <span className="badge-separator">•</span>
          <span>{totalAttacks.toLocaleString()} Total Attack{sortedAttacks.length !== 1 ? 's' : ''}</span>
        </div>
      </div>

      <div className="attack-list">
        {sortedAttacks.map((attack, index) => {
          const info = getAttackInfo(attack.name);
          const isExpanded = expandedAttacks.has(attack.name);
          const percentage = ((attack.count / totalAttacks) * 100).toFixed(1);

          return (
            <div
              key={index}
              className={`attack-card ${info.severity.toLowerCase()}`}
              style={{ borderLeftColor: info.color }}
            >
              <div className="attack-card-header" onClick={() => toggleExpand(attack.name)}>
                <div className="attack-header-left">
                  <span className="attack-icon">{info.icon}</span>
                  <div className="attack-title-section">
                    <h4 className="attack-name">{info.name}</h4>
                    <div className="attack-meta">
                      <span className="attack-category">{info.category}</span>
                      <span className="attack-severity" style={{ color: info.color }}>
                        {info.severity} Severity
                      </span>
                    </div>
                  </div>
                </div>
                <div className="attack-header-right">
                  <div className="attack-count-section">
                    <span className="attack-count">{attack.count.toLocaleString()}</span>
                    <span className="attack-percentage">{percentage}%</span>
                  </div>
                  <button className="expand-button">
                    {isExpanded ? '▼' : '▶'}
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div className="attack-details-content">
                  <div className="details-section">
                    <h5>📋 Description</h5>
                    <p>{info.description}</p>
                  </div>

                  <div className="details-section">
                    <h5>⚙️ How It Works</h5>
                    <p>{info.howItWorks}</p>
                  </div>

                  <div className="details-section">
                    <h5>💥 Impact</h5>
                    <p>{info.impact}</p>
                  </div>

                  <div className="details-section">
                    <h5>🛡️ Prevention Tips</h5>
                    <ul className="prevention-list">
                      {info.preventionTips.map((tip, tipIndex) => (
                        <li key={tipIndex}>
                          <span className="tip-icon">✓</span>
                          {tip}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="details-section">
                    <h5>🔍 Detection Signs</h5>
                    <ul className="detection-list">
                      {info.detectionSigns.map((sign, signIndex) => (
                        <li key={signIndex}>
                          <span className="sign-icon">⚠</span>
                          {sign}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="attack-visual">
                    <div className="visual-bar">
                      <div
                        className="visual-bar-fill"
                        style={{
                          width: `${percentage}%`,
                          backgroundColor: info.color
                        }}
                      ></div>
                    </div>
                    <div className="visual-stats">
                      <span>Count: {attack.count.toLocaleString()}</span>
                      <span>Percentage: {percentage}%</span>
                      <span>Severity: {info.severity}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="security-recommendations">
        <h4>🔒 General Security Recommendations</h4>
        <div className="recommendations-grid">
          <div className="recommendation-card">
            <span className="rec-icon">🛡️</span>
            <h5>Network Security</h5>
            <ul>
              <li>Implement firewalls</li>
              <li>Use IDS/IPS systems</li>
              <li>Network segmentation</li>
            </ul>
          </div>
          <div className="recommendation-card">
            <span className="rec-icon">🔐</span>
            <h5>Access Control</h5>
            <ul>
              <li>Strong passwords</li>
              <li>Multi-factor authentication</li>
              <li>Least privilege access</li>
            </ul>
          </div>
          <div className="recommendation-card">
            <span className="rec-icon">📊</span>
            <h5>Monitoring</h5>
            <ul>
              <li>Continuous monitoring</li>
              <li>Log analysis</li>
              <li>Alert systems</li>
            </ul>
          </div>
          <div className="recommendation-card">
            <span className="rec-icon">🔄</span>
            <h5>Maintenance</h5>
            <ul>
              <li>Regular updates</li>
              <li>Security patches</li>
              <li>Vulnerability assessments</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

