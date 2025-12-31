import React, { useState } from 'react';
import { FileUploader } from './components/FileUploader';
import { DatasetInfo } from './components/DatasetInfo';
import { SinglePrediction } from './components/SinglePrediction';
import { DetectionSummary } from './components/DetectionSummary';
import { AttackBreakdown } from './components/AttackBreakdown';
import { BinaryMetrics } from './components/BinaryMetrics';
import { MetricsExplainer } from './components/MetricsExplainer';
import { LiveMonitor } from './components/LiveMonitor';
import { RealTimeStats } from './components/RealTimeStats';
import { DashboardOverview } from './components/DashboardOverview';
import { AttackDetails } from './components/AttackDetails';
import { IntroPage } from './components/IntroPage';
import { NetworkAnalysis } from './components/NetworkAnalysis';
import { SettingsDashboard } from './components/SettingsDashboard';
import './App.css';

function App() {
  const [showIntro, setShowIntro] = useState(true);
  const [predictions, setPredictions] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [realtimeStats, setRealtimeStats] = useState(null);
  const [showLiveMonitor, setShowLiveMonitor] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleFileProcess = (data) => {
    setPredictions(data);
    setActiveTab('overview');
  };

  const handleDownload = () => {
    if (predictions?.download_url) {
      const link = document.createElement('a');
      link.href = predictions.download_url;
      link.download = 'ids_predictions.csv';
      link.click();
    }
  };

  if (showIntro) {
    return <IntroPage onEnter={() => setShowIntro(false)} />;
  }

  return (
    <div className="app-container">

      {/* NAVBAR */}
      <nav className="navbar">
        <div className="navbar-left">
          <button
            className="mobile-menu-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle menu"
          >
            ☰
          </button>
          <div className="navbar-logo">IDS</div>
          <h2>Intrusion Detection System (Keep your system safe)</h2>
          <span className="navbar-status">
            <span className="status-dot"></span> LIVE
          </span>
        </div>
      </nav>

      <div className="dashboard-layout">

        {/* SIDEBAR */}
        <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
          <button
            className="sidebar-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            ×
          </button>

          <div className="sidebar-section">
            <div className="sidebar-title">Menu</div>
            <button
              className={`sidebar-link ${activeTab === 'dashboard' && !showLiveMonitor ? 'active' : ''}`}
              onClick={() => { setActiveTab('dashboard'); setShowLiveMonitor(false); }}
            >
              <span className="sidebar-icon">📊</span> Dashboard
            </button>
            <button
              className={`sidebar-link ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => { setActiveTab('upload'); setShowLiveMonitor(false); }}
            >
              <span className="sidebar-icon">📤</span> Upload & Analyze
            </button>
            <button
              className={`sidebar-link ${showLiveMonitor ? 'active' : ''}`}
              onClick={() => { setShowLiveMonitor(true); setActiveTab('live'); }}
            >
              <span className="sidebar-icon">👁</span> Live Monitor
            </button>
            <button
              className={`sidebar-link ${activeTab === 'threats' ? 'active' : ''}`}
              onClick={() => { setActiveTab('threats'); setShowLiveMonitor(false); }}
            >
              <span className="sidebar-icon">⚠</span> Threats
            </button>
            <button
              className={`sidebar-link ${activeTab === 'network' ? 'active' : ''}`}
              onClick={() => { setActiveTab('network'); setShowLiveMonitor(false); }}
            >
              <span className="sidebar-icon">🌐</span> Network
            </button>
          </div>


          <div className="sidebar-section">
            <div className="sidebar-title">System</div>
            <button
              className={`sidebar-link ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => { setActiveTab('settings'); setShowLiveMonitor(false); }}
            >
              <span className="sidebar-icon">⚙</span> Settings
            </button>
          </div>

        </aside>

        {/* MAIN CONTENT */}
        <main className="main-container">
          <RealTimeStats onStatsUpdate={setRealtimeStats} />

          {showLiveMonitor ? (
            <LiveMonitor />
          ) : activeTab === 'settings' ? (
            <SettingsDashboard />
          ) : activeTab === 'network' ? (
            <NetworkAnalysis />
          ) : activeTab === 'dashboard' ? (
            <>
              <DashboardOverview />
              {realtimeStats && (
                <div className="realtime-stats-banner">
                  <div className="realtime-stat-item">
                    <span className="realtime-stat-label">Live Packets:</span>
                    <span className="realtime-stat-value">{realtimeStats.total_packets.toLocaleString()}</span>
                  </div>
                  <div className="realtime-stat-item">
                    <span className="realtime-stat-label">Threats:</span>
                    <span className="realtime-stat-value threat">{realtimeStats.threats_detected.toLocaleString()}</span>
                  </div>
                  <div className="realtime-stat-item">
                    <span className="realtime-stat-label">Critical Alerts:</span>
                    <span className="realtime-stat-value critical">{realtimeStats.critical_alerts.toLocaleString()}</span>
                  </div>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="header-section">
                <h1>Cybersecurity Threat Dashboard</h1>
                <p>
                  Real-time monitoring and intrusion detection system. Upload data to begin analysis.
                </p>
              </div>

              {realtimeStats && (
                <div className="realtime-stats-banner">
                  <div className="realtime-stat-item">
                    <span className="realtime-stat-label">Live Packets:</span>
                    <span className="realtime-stat-value">{realtimeStats.total_packets.toLocaleString()}</span>
                  </div>
                  <div className="realtime-stat-item">
                    <span className="realtime-stat-label">Threats:</span>
                    <span className="realtime-stat-value threat">{realtimeStats.threats_detected.toLocaleString()}</span>
                  </div>
                  <div className="realtime-stat-item">
                    <span className="realtime-stat-label">Critical Alerts:</span>
                    <span className="realtime-stat-value critical">{realtimeStats.critical_alerts.toLocaleString()}</span>
                  </div>
                </div>
              )}

              <DatasetInfo />

              {activeTab === 'upload' && <FileUploader onFileProcess={handleFileProcess} />}

              {/* TABS */}
              {predictions && (
                <div className="tabs">
                  <button
                    className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                  >
                    📊 Overview
                  </button>

                  <button
                    className={`tab ${activeTab === 'threats' ? 'active' : ''}`}
                    onClick={() => setActiveTab('threats')}
                  >
                    ⚠ Threats
                  </button>

                  {predictions.metrics && (
                    <button
                      className={`tab ${activeTab === 'metrics' ? 'active' : ''}`}
                      onClick={() => setActiveTab('metrics')}
                    >
                      📈 Metrics
                    </button>
                  )}
                </div>
              )}

              {/* TAB CONTENT */}
              {predictions && (
                <>
                  {activeTab === 'overview' && (
                    <div className="tab-content">
                      {predictions.single ? (
                        <SinglePrediction
                          prediction={predictions.pred_type}
                          probability={predictions.prob_attack}
                          topClasses={predictions.top_classes}
                        />
                      ) : (
                        <>
                          <DetectionSummary stats={predictions.stats} />
                          <AttackBreakdown attackCounts={predictions.attack_counts} />
                          <AttackDetails attackCounts={predictions.attack_counts} />
                        </>
                      )}
                    </div>
                  )}

                  {activeTab === 'threats' && (
                    <div className="tab-content">
                      <AttackBreakdown attackCounts={predictions.attack_counts} />
                      <AttackDetails attackCounts={predictions.attack_counts} />
                    </div>
                  )}

                  {activeTab === 'metrics' && predictions.metrics && (
                    <div className="tab-content">
                      <BinaryMetrics
                        metrics={predictions.metrics}
                        confusion={predictions.confusion}
                      />
                      <MetricsExplainer />
                    </div>
                  )}

                  {/* DOWNLOAD SECTION */}
                  <div className="card download-card">
                    <div className="download-header">
                      <h3>Export Results</h3>
                      <p className="download-subtitle">Download predictions in CSV format</p>
                    </div>

                    <button onClick={handleDownload} className="btn-download">
                      ↓ Download CSV
                    </button>
                  </div>
                </>
              )}
            </>
          )}

        </main>
      </div>
    </div>
  );
}

export default App;
