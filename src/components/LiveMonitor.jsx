import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import './LiveMonitor.css';

export function LiveMonitor() {
  const [connected, setConnected] = useState(false);
  const [monitoring, setMonitoring] = useState(false);
  const [stats, setStats] = useState({
    total_packets: 0,
    threats_detected: 0,
    critical_alerts: 0,
    defense_actions: 0,
    attack_types: {},
    recent_threats: []
  });
  const [latestFlow, setLatestFlow] = useState(null);
  const [packetHistory, setPacketHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const socketRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket server
    socketRef.current = io('http://localhost:5000', {
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      setConnected(true);
      console.log('Connected to real-time monitoring server');
    });

    socketRef.current.on('disconnect', () => {
      setConnected(false);
      setMonitoring(false);
      console.log('Disconnected from server');
    });

    socketRef.current.on('connected', (data) => {
      if (data.stats) {
        setStats(data.stats);
      }
      if (data.is_monitoring !== undefined) {
        setMonitoring(data.is_monitoring);
      }
    });

    socketRef.current.on('realtime_update', (data) => {
      if (data.stats) {
        setStats(data.stats);
      }
      if (data.latest_flow) {
        setLatestFlow(data.latest_flow);
        // Show all packets (up to 1000 for stability)
        setPacketHistory(prev => [data.latest_flow, ...prev].slice(0, 1000));
      }
    });

    socketRef.current.on('critical_alert', (alert) => {
      setAlerts(prev => [alert, ...prev].slice(0, 10));
      // Show browser notification if permission granted
      if (Notification.permission === 'granted') {
        new Notification('Critical Threat Detected', {
          body: `${alert.type} detected with ${(alert.probability * 100).toFixed(1)}% probability`,
          icon: '/favicon.ico'
        });
      }
    });

    socketRef.current.on('monitoring_started', () => {
      setMonitoring(true);
    });

    socketRef.current.on('monitoring_stopped', () => {
      setMonitoring(false);
    });

    socketRef.current.on('stats_reset', (data) => {
      if (data.stats) {
        setStats(data.stats);
      }
      setPacketHistory([]); // Clear local history
      setLatestFlow(null);
    });

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const handleStartMonitoring = () => {
    if (socketRef.current && connected) {
      socketRef.current.emit('start_monitoring');
    }
  };

  const handleStopMonitoring = () => {
    if (socketRef.current && connected) {
      socketRef.current.emit('stop_monitoring');
    }
  };

  const handleResetStats = () => {
    if (socketRef.current && connected) {
      socketRef.current.emit('reset_stats');
    }
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <div className="live-monitor">
      <div className="monitor-header">
        <h2>🔴 Live Network Monitoring</h2>
        <div className="monitor-controls">
          <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          {connected && (
            <>
              {!monitoring ? (
                <button className="btn-start" onClick={handleStartMonitoring}>
                  ▶ Start Monitoring
                </button>
              ) : (
                <button className="btn-stop" onClick={handleStopMonitoring}>
                  ⏸ Stop Monitoring
                </button>
              )}
              <button className="btn-reset" onClick={handleResetStats}>
                🔄 Reset Stats
              </button>
            </>
          )}
        </div>
      </div>

      {!connected && (
        <div className="connection-warning">
          ⚠️ Not connected to monitoring server. Make sure the backend is running on port 5000.
        </div>
      )}

      {connected && monitoring && (
        <div className="realtime-stats-grid">
          <div className="stat-card">
            <div className="stat-label">Total Packets</div>
            <div className="stat-value">{stats.total_packets.toLocaleString()}</div>
            <div className="stat-change">↑ Real-time</div>
          </div>

          <div className="stat-card threat">
            <div className="stat-label">Threats Detected</div>
            <div className="stat-value">{stats.threats_detected.toLocaleString()}</div>
            <div className="stat-change">Active monitoring</div>
          </div>

          <div className="stat-card critical">
            <div className="stat-label">Critical Alerts</div>
            <div className="stat-value">{stats.critical_alerts.toLocaleString()}</div>
            <div className="stat-change">Requires attention</div>
          </div>

          <div className="stat-card">
            <div className="stat-label">Defense Actions</div>
            <div className="stat-value">{stats.defense_actions.toLocaleString()}</div>
            <div className="stat-change">Auto-defense active</div>
          </div>
        </div>
      )}

      {latestFlow && monitoring && (
        <div className="latest-flow">
          <h3>Latest Flow Analysis</h3>
          <div className="flow-details">
            {latestFlow.source_ip && (
              <div className="flow-item">
                <span className="flow-label">Source:</span>
                <span className="flow-value small">{latestFlow.source_ip}</span>
              </div>
            )}
            {latestFlow.dest_ip && (
              <div className="flow-item">
                <span className="flow-label">Dest:</span>
                <span className="flow-value small">{latestFlow.dest_ip}</span>
              </div>
            )}
            {latestFlow.protocol && (
              <div className="flow-item">
                <span className="flow-label">Protocol:</span>
                <span className="flow-value">{latestFlow.protocol}</span>
              </div>
            )}
            <div className="flow-item">
              <span className="flow-label">Prediction:</span>
              <span className={`flow-value ${latestFlow.is_attack ? 'attack' : 'benign'}`}>
                {latestFlow.prediction}
              </span>
            </div>
            {latestFlow.probability !== null && (
              <div className="flow-item">
                <span className="flow-label">Attack Probability:</span>
                <span className="flow-value">
                  {(latestFlow.probability * 100).toFixed(2)}%
                </span>
              </div>
            )}
            <div className="flow-item">
              <span className="flow-label">Timestamp:</span>
              <span className="flow-value">{formatTime(latestFlow.timestamp)}</span>
            </div>
          </div>
        </div>
      )}

      {stats.attack_types && Object.keys(stats.attack_types).length > 0 && (
        <div className="attack-types">
          <h3>Attack Type Distribution</h3>
          <div className="attack-list">
            {Object.entries(stats.attack_types)
              .sort((a, b) => b[1] - a[1])
              .map(([type, count]) => (
                <div key={type} className="attack-item">
                  <span className="attack-type">{type}</span>
                  <span className="attack-count">{count}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {alerts.length > 0 && (
        <div className="critical-alerts">
          <h3>🚨 Critical Alerts</h3>
          <div className="alerts-list">
            {alerts.map((alert, idx) => (
              <div key={idx} className="alert-item critical">
                <div className="alert-header">
                  <span className="alert-type">{alert.type}</span>
                  <span className="alert-severity">{alert.severity}</span>
                </div>
                <div className="alert-details">
                  <span>Probability: {(alert.probability * 100).toFixed(1)}%</span>
                  <span>{formatTime(alert.timestamp)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live Packet Log (Wireshark-like) */}
      {(monitoring || packetHistory.length > 0) && (
        <div className="packet-log">
          <h3>📡 Live Packet Capture</h3>
          <div className="table-container">
            <table className="packet-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Source</th>
                  <th>Destination</th>
                  <th>Protocol</th>
                  <th>Length</th>
                  <th>Info (Prediction)</th>
                </tr>
              </thead>
              <tbody>
                {/* List all captured packets (up to 1000) */}
                {packetHistory.map((pkt, idx) => (
                  <tr key={idx} className={pkt.is_attack ? 'row-attack' : 'row-benign'}>
                    <td>{formatTime(pkt.timestamp)}</td>
                    <td>{pkt.source_ip || '-'}</td>
                    <td>{pkt.dest_ip || '-'}</td>
                    <td>{pkt.protocol || 'TCP'}</td>
                    <td>{pkt.length || '0'}</td>
                    <td>{pkt.prediction} ({pkt.is_attack ? 'THREAT' : 'Safe'})</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {stats.recent_threats && stats.recent_threats.length > 0 && (
        <div className="recent-threats">
          <h3>Recent Threats</h3>
          <div className="threats-list">
            {stats.recent_threats.slice(-10).reverse().map((threat) => (
              <div key={threat.id} className={`threat-item ${threat.severity}`}>
                <div className="threat-header">
                  <span className="threat-type">{threat.type}</span>
                  <span className="threat-probability">
                    {(threat.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="threat-time">{formatTime(threat.timestamp)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}








