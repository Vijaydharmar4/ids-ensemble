import React from 'react';
import './NetworkArchitecture.css';

export function NetworkArchitecture() {
    return (
        <div className="architecture-container">
            <div className="arch-header">
                <h3>🛡️ System Architecture Overview</h3>
                <p>Logical flow of traffic inspection and component interaction</p>
            </div>

            <div className="arch-diagram">
                {/* External World */}
                <div className="arch-zone external">
                    <div className="arch-node internet">
                        <span className="arch-icon">☁️</span>
                        <span>Internet</span>
                    </div>
                </div>

                <div className="flow-arrow">⬇️ Traffic</div>

                {/* Perimeter Defense */}
                <div className="arch-zone perimeter">
                    <div className="arch-node firewall">
                        <span className="arch-icon">🔥</span>
                        <span>Firewall</span>
                    </div>
                    <div className="arch-node ids active">
                        <span className="arch-icon">👁️</span>
                        <span>IDS Engine</span>
                        <div className="pulse-indicator"></div>
                    </div>
                </div>

                <div className="flow-arrow">⬇️ Filtered</div>

                {/* Internal Network */}
                <div className="arch-zone internal">
                    <div className="arch-node switch">
                        <span className="arch-icon">🔄</span>
                        <span>Switch</span>
                    </div>

                    <div className="internal-nodes">
                        <div className="arch-node device">Admin PC</div>
                        <div className="arch-node device">Web Server</div>
                        <div className="arch-node device">Database</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
