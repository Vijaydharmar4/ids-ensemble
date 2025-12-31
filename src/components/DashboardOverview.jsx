import React from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import './DashboardOverview.css';

const COLORS = ['#4a9eff', '#ff6b6b', '#ff9800', '#4caf50', '#9c27b0', '#00bcd4', '#ffeb3b', '#795548'];

// Preloaded dataset statistics
const datasetStats = {
  totalFlows: 2830743,
  benignFlows: 2273097,
  attackFlows: 557646,
  features: 84,
  attackClasses: 14,
  timePeriod: '5 Days'
};

const attackDistribution = [
  { name: 'Benign', value: 2273097, percentage: 80.3, color: '#4a9eff' },
  { name: 'DoS Hulk', value: 231073, percentage: 8.2, color: '#ff6b6b' },
  { name: 'PortScan', value: 158930, percentage: 5.6, color: '#ff9800' },
  { name: 'DDoS', value: 128027, percentage: 4.5, color: '#f44336' },
  { name: 'DoS GoldenEye', value: 10293, percentage: 0.4, color: '#e91e63' },
  { name: 'FTP-Brute Force', value: 7938, percentage: 0.3, color: '#9c27b0' },
  { name: 'SSH-Brute Force', value: 5897, percentage: 0.2, color: '#673ab7' },
  { name: 'DoS Slowloris', value: 5796, percentage: 0.2, color: '#3f51b5' },
  { name: 'DoS Slowhttptest', value: 5499, percentage: 0.2, color: '#2196f3' },
  { name: 'Bot', value: 1966, percentage: 0.1, color: '#00bcd4' },
  { name: 'Web Attack', value: 2180, percentage: 0.1, color: '#009688' },
  { name: 'Infiltration', value: 36, percentage: 0.0, color: '#4caf50' },
  { name: 'Heartbleed', value: 11, percentage: 0.0, color: '#8bc34a' }
];

const attackTypesBar = attackDistribution
  .filter(item => item.name !== 'Benign')
  .map(item => ({
    name: item.name,
    count: item.value,
    percentage: item.percentage
  }))
  .sort((a, b) => b.count - a.count);

const protocolDistribution = [
  { protocol: 'TCP', count: 2150000, percentage: 76.0 },
  { protocol: 'UDP', count: 520000, percentage: 18.4 },
  { protocol: 'ICMP', count: 160743, percentage: 5.6 }
];

const dailyDistribution = [
  { day: 'Monday', flows: 450000, attacks: 85000 },
  { day: 'Tuesday', flows: 520000, attacks: 95000 },
  { day: 'Wednesday', flows: 580000, attacks: 110000 },
  { day: 'Thursday', flows: 610000, attacks: 125000 },
  { day: 'Friday', flows: 670743, attacks: 142646 }
];

const featureCategories = [
  { category: 'Flow Duration', count: 5, description: 'Time-based features' },
  { category: 'Packet Statistics', count: 25, description: 'Packet counts and sizes' },
  { category: 'Byte Statistics', count: 15, description: 'Byte counts and ratios' },
  { category: 'Flag Statistics', count: 8, description: 'TCP flag information' },
  { category: 'Rate Features', count: 12, description: 'Error and service rates' },
  { category: 'Host Statistics', count: 10, description: 'Host-based features' },
  { category: 'Protocol Info', count: 9, description: 'Protocol and service data' }
];

export function DashboardOverview() {
  return (
    <div className="dashboard-overview">
      <div className="overview-header">
        <h1>📊 Dataset Overview Dashboard</h1>
        <p className="subtitle">CICIDS2017 - Comprehensive Network Traffic Analysis</p>
      </div>

      {/* Key Statistics Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📦</div>
          <div className="stat-content">
            <div className="stat-label">Total Network Flows</div>
            <div className="stat-value">{datasetStats.totalFlows.toLocaleString()}</div>
            <div className="stat-desc">Analyzed network connections</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">✅</div>
          <div className="stat-content">
            <div className="stat-label">Benign Traffic</div>
            <div className="stat-value">{datasetStats.benignFlows.toLocaleString()}</div>
            <div className="stat-desc">{((datasetStats.benignFlows / datasetStats.totalFlows) * 100).toFixed(1)}% of total</div>
          </div>
        </div>

        <div className="stat-card threat">
          <div className="stat-icon">⚠️</div>
          <div className="stat-content">
            <div className="stat-label">Attack Traffic</div>
            <div className="stat-value">{datasetStats.attackFlows.toLocaleString()}</div>
            <div className="stat-desc">{((datasetStats.attackFlows / datasetStats.totalFlows) * 100).toFixed(1)}% of total</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🔢</div>
          <div className="stat-content">
            <div className="stat-label">Features</div>
            <div className="stat-value">{datasetStats.features}</div>
            <div className="stat-desc">Network flow characteristics</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🎯</div>
          <div className="stat-content">
            <div className="stat-label">Attack Classes</div>
            <div className="stat-value">{datasetStats.attackClasses}</div>
            <div className="stat-desc">Different attack types</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">📅</div>
          <div className="stat-content">
            <div className="stat-label">Time Period</div>
            <div className="stat-value">{datasetStats.timePeriod}</div>
            <div className="stat-desc">Data collection duration</div>
          </div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="charts-row">
        <div className="chart-card">
          <h3>🔄 Attack vs Benign Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Benign', value: datasetStats.benignFlows },
                  { name: 'Attacks', value: datasetStats.attackFlows }
                ]}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                <Cell fill="#4a9eff" />
                <Cell fill="#ff6b6b" />
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          <p className="chart-description">
            The dataset contains {((datasetStats.benignFlows / datasetStats.totalFlows) * 100).toFixed(1)}% benign traffic 
            and {((datasetStats.attackFlows / datasetStats.totalFlows) * 100).toFixed(1)}% attack traffic, 
            providing a balanced representation for training intrusion detection models.
          </p>
        </div>

        <div className="chart-card">
          <h3>📊 Attack Type Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={attackTypesBar.slice(0, 8)}>
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip formatter={(value) => value.toLocaleString()} />
              <Bar dataKey="count" fill="#ff6b6b" />
            </BarChart>
          </ResponsiveContainer>
          <p className="chart-description">
            Top attack types detected in the dataset. DoS Hulk and PortScan attacks are the most prevalent, 
            representing {attackDistribution[1].percentage}% and {attackDistribution[2].percentage}% of total attacks respectively.
          </p>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="charts-row">
        <div className="chart-card">
          <h3>🌐 Protocol Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={protocolDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ protocol, percentage }) => `${protocol}: ${percentage}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="count"
              >
                {protocolDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
          <p className="chart-description">
            Network protocols used in the captured traffic. TCP dominates with {protocolDistribution[0].percentage}% of flows, 
            followed by UDP ({protocolDistribution[1].percentage}%) and ICMP ({protocolDistribution[2].percentage}%).
          </p>
        </div>

        <div className="chart-card">
          <h3>📈 Daily Traffic & Attack Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dailyDistribution}>
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="flows" stroke="#4a9eff" name="Total Flows" strokeWidth={2} />
              <Line type="monotone" dataKey="attacks" stroke="#ff6b6b" name="Attacks" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <p className="chart-description">
            Traffic volume and attack frequency over the 5-day collection period. 
            Both metrics show an increasing trend, with Friday experiencing the highest activity.
          </p>
        </div>
      </div>

      {/* Feature Categories */}
      <div className="chart-card full-width">
        <h3>🔍 Feature Categories Breakdown</h3>
        <div className="feature-grid">
          {featureCategories.map((category, index) => (
            <div key={index} className="feature-item">
              <div className="feature-header">
                <span className="feature-name">{category.category}</span>
                <span className="feature-count">{category.count} features</span>
              </div>
              <p className="feature-desc">{category.description}</p>
              <div className="feature-bar">
                <div 
                  className="feature-bar-fill" 
                  style={{ width: `${(category.count / 84) * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
        <p className="chart-description">
          The dataset includes {datasetStats.features} features organized into {featureCategories.length} main categories, 
          covering flow duration, packet statistics, byte counts, TCP flags, error rates, host information, and protocol details.
        </p>
      </div>

      {/* Attack Types Detailed */}
      <div className="chart-card full-width">
        <h3>🎯 Complete Attack Type Breakdown</h3>
        <div className="attack-types-grid">
          {attackDistribution.map((attack, index) => (
            <div key={index} className={`attack-type-card ${attack.name === 'Benign' ? 'benign' : 'attack'}`}>
              <div className="attack-type-header">
                <div className="attack-color" style={{ backgroundColor: attack.color }}></div>
                <div className="attack-info">
                  <div className="attack-name">{attack.name}</div>
                  <div className="attack-count">{attack.value.toLocaleString()} flows</div>
                </div>
                <div className="attack-percentage">{attack.percentage}%</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Information */}
      <div className="info-sections">
        <div className="info-card">
          <h3>📚 Dataset Description</h3>
          <p>
            The CICIDS2017 dataset is a comprehensive benchmark dataset for intrusion detection systems. 
            It contains network traffic flows captured over 5 days, including both normal (benign) traffic 
            and various attack scenarios. The dataset was created using the CICFlowMeter tool to extract 
            flow-based features from network packets.
          </p>
          <p>
            This dataset is widely used in cybersecurity research for training and evaluating machine learning 
            models for intrusion detection. It provides realistic attack scenarios including DoS, DDoS, 
            port scanning, brute force attacks, web attacks, and infiltration attempts.
          </p>
        </div>

        <div className="info-card">
          <h3>💡 Key Insights</h3>
          <ul className="insights-list">
            <li>✅ <strong>Balanced Dataset:</strong> {((datasetStats.benignFlows / datasetStats.totalFlows) * 100).toFixed(1)}% benign traffic ensures realistic model training</li>
            <li>⚠️ <strong>Attack Diversity:</strong> {datasetStats.attackClasses} different attack types provide comprehensive coverage</li>
            <li>📊 <strong>Large Scale:</strong> Over {Math.floor(datasetStats.totalFlows / 1000000)}M flows enable robust model training</li>
            <li>🔍 <strong>Rich Features:</strong> {datasetStats.features} features capture comprehensive network behavior</li>
            <li>🌐 <strong>Real-world Scenarios:</strong> Captured in controlled lab environment simulating real attacks</li>
            <li>📈 <strong>Temporal Data:</strong> 5-day collection period shows traffic patterns over time</li>
          </ul>
        </div>
      </div>
    </div>
  );
}








