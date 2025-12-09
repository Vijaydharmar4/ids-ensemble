import React from 'react';
import { normalizeAttackName, explainAttackType } from '../utils/helpers';
import './SinglePrediction.css';

export function SinglePrediction({ prediction, probability, topClasses }) {
  const predType = prediction;
  const predAttack = normalizeAttackName(predType) !== 'benign' ? 'ATTACK' : 'BENIGN';
  const isAttack = predAttack === 'ATTACK';

  return (
    <div className="card single-prediction-card">
      <h3>🧪 Single Record Prediction</h3>
      
      <div className="prediction-result">
        <div className={`prediction-badge ${isAttack ? 'attack' : 'benign'}`}>
          {predAttack}
        </div>
        <div style={{ marginTop: '20px' }}>
          <strong style={{ color: '#90a4ae', fontSize: '1rem' }}>Attack Type:</strong>
          <code style={{ marginLeft: '10px', fontSize: '1.1rem' }}>{predType}</code>
        </div>
      </div>

      {probability !== null && (
        <div className="probability-display">
          <div style={{ color: '#90a4ae', fontSize: '0.9rem', marginBottom: '8px' }}>
            Attack Probability
          </div>
          <div className="probability-value">
            {(probability * 100).toFixed(2)}%
          </div>
        </div>
      )}

      {topClasses && (
        <div style={{ 
          marginTop: '30px', 
          padding: '20px', 
          background: 'rgba(100, 200, 255, 0.05)', 
          borderRadius: '12px',
          border: '1px solid rgba(100, 200, 255, 0.1)'
        }}>
          <h4 style={{ color: '#64b5f6', marginBottom: '15px', fontSize: '1.1rem' }}>
            Top Predictions
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {topClasses.map((c, idx) => (
              <div key={idx} style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '10px',
                background: 'rgba(100, 200, 255, 0.05)',
                borderRadius: '8px'
              }}>
                <span style={{ color: '#e0e0e0', fontWeight: '600' }}>{c.class}</span>
                <span style={{ color: '#64b5f6', fontWeight: '700' }}>
                  {(c.prob * 100).toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {isAttack && (
        <div className="info-box" style={{ marginTop: '30px', textAlign: 'left' }}>
          <h4 style={{ color: '#ff6b6b', marginBottom: '10px' }}>⚠️ Attack Details</h4>
          <p>{explainAttackType(predType)}</p>
        </div>
      )}
    </div>
  );
}