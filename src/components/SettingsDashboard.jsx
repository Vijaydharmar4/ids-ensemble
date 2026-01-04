import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import './SettingsDashboard.css';

export function SettingsDashboard() {
  const [notifications, setNotifications] = useState(true);
  const [sensitivity, setSensitivity] = useState(80);
  const [retention, setRetention] = useState(30);
  const [forceAttackMode, setForceAttackMode] = useState(false);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5000', {
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      setConnected(true);
    });

    socketRef.current.on('connected', (data) => {
      if (data.force_attack_mode !== undefined) {
        setForceAttackMode(data.force_attack_mode);
      }
    });

    socketRef.current.on('attack_mode_toggled', (data) => {
      setForceAttackMode(data.enabled);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const toggleForceAttackMode = () => {
    if (socketRef.current && connected) {
      socketRef.current.emit('toggle_attack_mode', { enabled: !forceAttackMode });
    }
  };

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h2>⚙ System Configuration</h2>
        <p>Manage global parameters and user preferences</p>
      </div>

      <div className="settings-grid">
        {/* Security Policies */}
        <div className="settings-card">
          <h3>🛡️ Detection Policies</h3>

          <div className="setting-item">
            <div className="setting-label">
              <span>Threat Sensitivity Threshold</span>
              <span className="value">{sensitivity}%</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              value={sensitivity}
              onChange={(e) => setSensitivity(e.target.value)}
              className="range-input"
            />
            <p className="setting-help">Alerts triggered only when confidence exceeds this value.</p>
          </div>

          <div className="setting-item">
            <div className="setting-label">
              <span>Log Retention Period</span>
              <span className="value">{retention} Days</span>
            </div>
            <input
              type="range"
              min="1"
              max="365"
              value={retention}
              onChange={(e) => setRetention(e.target.value)}
              className="range-input"
            />
          </div>
        </div>

        {/* System Preferences */}
        <div className="settings-card">
          <h3>🔔 Notifications & Alerts</h3>

          <div className="setting-row">
            <div>
              <div className="setting-title">Real-time Popup Alerts</div>
              <div className="setting-help">Show immediate browser notifications for critical threats.</div>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={notifications}
                onChange={() => setNotifications(!notifications)}
              />
              <span className="slider round"></span>
            </label>
          </div>

          <div className="setting-row">
            <div>
              <div className="setting-title">Email Reports</div>
              <div className="setting-help">Receive daily summary reports via email.</div>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" />
              <span className="slider round"></span>
            </label>
          </div>
        </div>

        {/* Demonstration Settings */}
        <div className="settings-card highlight">
          <h3>🧪 Demonstration Settings</h3>
          <p className="section-desc" style={{ color: '#90a4ae', fontSize: '0.9rem', marginBottom: '15px' }}>
            Special tools for testing and verifying the system's detection capabilities.
          </p>

          <div className="setting-row">
            <div>
              <div className="setting-title" style={{ fontWeight: '600', color: '#eceff1' }}>Enable Forced Threats</div>
              <div className="setting-help" style={{ color: '#b0bec5', fontSize: '0.85rem', marginTop: '4px', maxWidth: '300px' }}>
                When enabled, the system will randomly inject threat packets (30% probability)
                into the live monitoring stream. Use this to demonstrate detection alerts and
                system responses during a presentation.
              </div>
            </div>
            <label className="toggle-switch danger">
              <input
                type="checkbox"
                checked={forceAttackMode}
                onChange={toggleForceAttackMode}
              />
              <span className="slider round"></span>
            </label>
          </div>
          {!connected && <div className="connection-error" style={{ color: '#ff4757', marginTop: '10px', fontSize: '0.85rem' }}>⚠️ Disconnected from monitoring backend</div>}
        </div>

        {/* System Info */}
        <div className="settings-card">
          <h3>ℹ️ About System</h3>
          <div className="info-row">
            <span>Version:</span>
            <span className="mono">v2.4.0-stable</span>
          </div>
          <div className="info-row">
            <span>Model Engine:</span>
            <span className="mono">CICIDS-Ensemble-X</span>
          </div>
          <div className="info-row">
            <span>Last Update:</span>
            <span className="mono">{new Date().toLocaleDateString()}</span>
          </div>

          <button className="btn-update">
            Check for Updates
          </button>
        </div>
      </div>
    </div>
  );
}
