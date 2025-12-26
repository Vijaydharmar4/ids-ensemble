import React from 'react';
import '../styles/DatasetInfo.css';

export function DatasetInfo() {
  return (
    <div className="dataset-info-card">
      <div className="dataset-header">
        <div className="dataset-icon">📊</div>
        <div className="dataset-title-section">
          <h2>Pre-loaded Dataset</h2>
          <p className="dataset-name">CICIDS2017 - Multiclass Attack Detection</p>
        </div>
      </div>

      <div className="dataset-content">
        <div className="info-section">
          <h3>Dataset Overview</h3>
          <p>
            The CICIDS2017 dataset is a benchmark cybersecurity dataset containing network traffic 
            flows from real-world attack scenarios. It combines benign and attack traffic with labeled 
            attack types, making it ideal for training and evaluating intrusion detection systems.
          </p>
        </div>

        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Total Flows</span>
            <span className="info-value">2,830,743</span>
          </div>
          <div className="info-item">
            <span className="info-label">Features</span>
            <span className="info-value">84</span>
          </div>
          <div className="info-item">
            <span className="info-label">Attack Classes</span>
            <span className="info-value">14</span>
          </div>
          <div className="info-item">
            <span className="info-label">Time Period</span>
            <span className="info-value">5 Days</span>
          </div>
        </div>

        <div className="info-section">
          <h3>Attack Classes</h3>
          <div className="attack-classes">
            <div className="attack-class">
              <span className="class-badge benign">Benign</span>
              <span className="class-desc">Normal, non-malicious traffic</span>
            </div>
            <div className="attack-class">
              <span className="class-badge dos">DoS Slowhttptest</span>
              <span className="class-desc">Denial of Service attack variant</span>
            </div>
            <div className="attack-class">
              <span className="class-badge dos">DoS Slowloris</span>
              <span className="class-desc">HTTP-based DoS attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge dos">DoS Hulk</span>
              <span className="class-desc">High-volume DoS attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge dos">DoS GoldenEye</span>
              <span className="class-desc">Layer 7 DoS attack tool</span>
            </div>
            <div className="attack-class">
              <span className="class-badge ddos">DDoS</span>
              <span className="class-desc">Distributed Denial of Service</span>
            </div>
            <div className="attack-class">
              <span className="class-badge probe">PortScan</span>
              <span className="class-desc">Network reconnaissance attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge probe">nmap</span>
              <span className="class-desc">Port scanning tool attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge brute">SSH-Brute Force</span>
              <span className="class-desc">Credential brute force attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge brute">FTP-Brute Force</span>
              <span className="class-desc">FTP credential attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge infiltration">Bot</span>
              <span className="class-desc">Botnet infiltration traffic</span>
            </div>
            <div className="attack-class">
              <span className="class-badge infiltration">Infiltration</span>
              <span className="class-desc">Network penetration attack</span>
            </div>
            <div className="attack-class">
              <span className="class-badge web">Web Attack</span>
              <span className="class-desc">HTTP/Web layer attacks</span>
            </div>
            <div className="attack-class">
              <span className="class-badge xml">XML-RPC</span>
              <span className="class-desc">XML-RPC vulnerability exploit</span>
            </div>
          </div>
        </div>

        <div className="info-section">
          <h3>Key Features</h3>
          <ul className="features-list">
            <li>Flow-based network traffic analysis</li>
            <li>Bidirectional flow information</li>
            <li>Statistical and time-series features</li>
            <li>Real attack traffic captured in lab environment</li>
            <li>Labeled attack types for supervised learning</li>
            <li>Multiple attack categories (DoS, Probe, R2L, U2R)</li>
            <li>Balanced class representation for fair evaluation</li>
            <li>CSV format for easy integration</li>
          </ul>
        </div>

        <div className="info-section">
          <h3>Dataset Structure</h3>
          <div className="structure-info">
            <div className="structure-item">
              <span className="structure-label">Flow Duration:</span>
              <span>Microseconds</span>
            </div>
            <div className="structure-item">
              <span className="structure-label">Payload Information:</span>
              <span>Forward/Backward packet data</span>
            </div>
            <div className="structure-item">
              <span className="structure-label">Packet Statistics:</span>
              <span>Lengths, inter-arrival times, flags</span>
            </div>
            <div className="structure-item">
              <span className="structure-label">Flow Statistics:</span>
              <span>Protocol, port numbers, IP addresses</span>
            </div>
          </div>
        </div>

        <div className="info-section">
          <h3>Use Cases</h3>
          <div className="use-cases">
            <div className="use-case-item">
              <span className="use-case-icon">🔍</span>
              <div>
                <strong>Model Training</strong>
                <p>Train machine learning models for intrusion detection</p>
              </div>
            </div>
            <div className="use-case-item">
              <span className="use-case-icon">📈</span>
              <div>
                <strong>Performance Evaluation</strong>
                <p>Benchmark IDS algorithms and techniques</p>
              </div>
            </div>
            <div className="use-case-item">
              <span className="use-case-icon">🛡️</span>
              <div>
                <strong>Security Research</strong>
                <p>Analyze attack patterns and network behavior</p>
              </div>
            </div>
            <div className="use-case-item">
              <span className="use-case-icon">⚙️</span>
              <div>
                <strong>System Optimization</strong>
                <p>Improve detection accuracy and reduce false positives</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}