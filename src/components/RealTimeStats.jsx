import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

export function RealTimeStats({ onStatsUpdate }) {
  const [stats, setStats] = useState({
    total_packets: 0,
    threats_detected: 0,
    critical_alerts: 0,
    defense_actions: 0
  });
  const socketRef = useRef(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5000', {
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      console.log('Real-time stats connected');
    });

    socketRef.current.on('realtime_update', (data) => {
      if (data.stats) {
        setStats(data.stats);
        if (onStatsUpdate) {
          onStatsUpdate(data.stats);
        }
      }
    });

    socketRef.current.on('connected', (data) => {
      if (data.stats) {
        setStats(data.stats);
        if (onStatsUpdate) {
          onStatsUpdate(data.stats);
        }
      }
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [onStatsUpdate]);

  return null; // This component just updates state, no UI
}








