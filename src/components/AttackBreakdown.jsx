import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, Cell } from 'recharts';
import './AttackBreakdown.css';

export function AttackBreakdown({ attackCounts }) {
  // Handle both object and array formats
  let chartData = [];
  
  if (!attackCounts) {
    chartData = [];
  } else if (Array.isArray(attackCounts)) {
    chartData = attackCounts;
  } else if (typeof attackCounts === 'object') {
    // Convert object to array format
    chartData = Object.entries(attackCounts)
      .map(([name, count]) => ({
        name,
        count: typeof count === 'number' ? count : parseInt(count) || 0
      }))
      .sort((a, b) => b.count - a.count);
  }

  if (chartData.length === 0) {
    return (
      <div className="card">
        <h3>🧭 Predicted Attack Types</h3>
        <div className="info-box">No attacks predicted in the uploaded data.</div>
      </div>
    );
  }

  const COLORS = ['#ff6b6b', '#ff9800', '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3'];
  const totalAttacks = chartData.reduce((sum, item) => sum + item.count, 0);

  return (
    <div className="card attack-breakdown-container">
      <h3>🧭 Predicted Attack Types</h3>
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={chartData} margin={{ top: 25, right: 30, left: 20, bottom: 70 }}>
            <XAxis 
              dataKey="name" 
              angle={-45}
              textAnchor="end"
              height={120}
              tick={{ fill: '#90a4ae', fontSize: 11, fontWeight: 600 }}
              interval={0}
            />
            <YAxis 
              tick={{ fill: '#90a4ae', fontSize: 12 }} 
              label={{ value: 'Count', angle: -90, position: 'insideLeft', fill: '#90a4ae' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(30, 42, 58, 0.98)', 
                border: '2px solid rgba(100, 200, 255, 0.4)',
                borderRadius: '12px',
                color: '#e0e0e0',
                padding: '12px',
                boxShadow: '0 8px 25px rgba(0, 0, 0, 0.4)'
              }}
              formatter={(value, name) => [
                `${value.toLocaleString()} attacks`,
                name
              ]}
              labelStyle={{ color: '#64b5f6', fontWeight: 700, marginBottom: '8px' }}
            />
            <Bar dataKey="count" radius={[10, 10, 0, 0]} barSize={60}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="attack-summary">
        <p>Total attack types detected: <strong>{chartData.length}</strong></p>
        <p>Total attack flows: <strong>{totalAttacks.toLocaleString()}</strong></p>
        <p>Average per type: <strong>{Math.round(totalAttacks / chartData.length).toLocaleString()}</strong></p>
      </div>
    </div>
  );
}