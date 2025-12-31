import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import { NetworkArchitecture } from './NetworkArchitecture';
import './NetworkAnalysis.css';

export function NetworkAnalysis() {
    const [topology, setTopology] = useState({ nodes: [], links: [] });
    const [activeLinks, setActiveLinks] = useState([]);
    const socketRef = useRef(null);

    // Fixed positions for known nodes to keep map stable
    const getNodePosition = (node) => {
        if (node.type === 'router') return { top: '10%', left: '50%' };
        // Bottom Row
        if (node.label === 'Admin PC') return { top: '85%', left: '20%' };
        if (node.label === 'Web Server') return { top: '85%', left: '50%' };
        if (node.label === 'IoT Device') return { top: '85%', left: '80%' };
        // Middle Row
        if (node.label === 'Workstation A') return { top: '45%', left: '80%' };
        if (node.label === 'DB Server') return { top: '45%', left: '20%' };
        return { top: '10%', left: '10%' }; // Fallback
    };

    useEffect(() => {
        socketRef.current = io('http://localhost:5000', {
            transports: ['websocket', 'polling']
        });

        socketRef.current.on('connect', () => {
            console.log("Network Map connected");
        });

        socketRef.current.on('realtime_update', (data) => {
            if (data.stats && data.stats.network_topology) {
                setTopology(data.stats.network_topology);

                // Visualize active links from the last few seconds
                const recent = data.stats.network_topology.links.slice(-5);
                setActiveLinks(recent);
            }
        });

        return () => {
            if (socketRef.current) socketRef.current.disconnect();
        };
    }, []);

    return (
        <div className="network-analysis">
            <div className="network-header">
                <h2>🌐 Real-Time Network Topology</h2>
                <div className="network-status">
                    <span className="status-badge active">Live Map</span>
                    <span className="status-badge secure">Active Nodes: {topology.nodes.length}</span>
                </div>
            </div>

            <NetworkArchitecture />

            <div className="network-grid">
                {/* Topology Map */}
                <div className="network-card topology">
                    <div className="topology-visual">
                        {/* Render Active Links as SVG Lines */}
                        <svg className="connections-overlay">
                            {activeLinks.map((link, i) => {
                                // Simple hack: We need absolute coords. 
                                // For now, we'll just draw lines between center and target logic if possible
                                // Or just animate the nodes themselves
                                return null;
                            })}
                            {/* Note: Drawing reliable lines between % sourced divs in React without a library like D3/React-Flow 
                   is complex. Instead, we will highlight the nodes that are talking. */}
                        </svg>

                        {/* Render Nodes */}
                        {topology.nodes.map((node, i) => {
                            const pos = getNodePosition(node);
                            // Check if this node is source or target of an active link
                            const isActive = activeLinks.some(l => l.source === node.ip || l.target === node.ip);

                            return (
                                <div
                                    key={i}
                                    className={`node ${node.type} ${isActive ? 'transmitting' : ''}`}
                                    style={{ top: pos.top, left: pos.left, position: 'absolute' }}
                                >
                                    <span className="icon">
                                        {node.type === 'router' ? '📡' :
                                            node.type === 'server' ? '🖥️' :
                                                node.type === 'iot' ? '📱' : '💻'}
                                    </span>
                                    <span className="label">{node.label}</span>
                                    <span className="ip">{node.ip}</span>
                                    {isActive && <div className="activity-pulse"></div>}
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Network Health */}
                <div className="network-card health">
                    <h3>Active Connections</h3>
                    <div className="active-connections-list">
                        {activeLinks.length === 0 && <p className="no-data">Waiting for traffic...</p>}
                        {activeLinks.map((link, i) => (
                            <div key={i} className="connection-item">
                                <span className="c-source">{link.source}</span>
                                <span className="c-arrow">→</span>
                                <span className="c-target">{link.target}</span>
                                <span className={`c-proto ${link.proto}`}>{link.proto}</span>
                            </div>
                        )).reverse()}
                    </div>
                </div>
            </div>
        </div>
    );
}
