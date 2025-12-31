# Real-Time Monitoring Setup Guide

This guide explains how to set up and use the real-time monitoring features of the Intrusion Detection System.

## Features

- **Real-time Network Monitoring**: Live packet analysis and threat detection
- **WebSocket Communication**: Instant updates without page refresh
- **Live Statistics**: Real-time dashboard showing packets, threats, and alerts
- **Critical Alert Notifications**: Browser notifications for high-severity threats
- **Attack Type Tracking**: Real-time breakdown of detected attack types

## Prerequisites

1. Python 3.7+
2. Node.js and npm
3. Flask backend dependencies
4. React frontend dependencies

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your model files are in the `artifacts/` directory:
   - `cicids_multiclass.joblib` (or other compatible models)

3. Start the Flask backend server:
```bash
python backend.py
```

The server will start on `http://localhost:5000` with WebSocket support.

### Frontend Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Start the React development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`.

## Usage

### Starting Real-Time Monitoring

1. Open the application in your browser
2. Click on "Live Monitor" in the sidebar
3. Click the "Start Monitoring" button
4. The system will begin generating and analyzing network flows in real-time

### Monitoring Features

- **Connection Status**: Shows whether you're connected to the monitoring server
- **Real-Time Statistics**: 
  - Total packets processed
  - Threats detected
  - Critical alerts
  - Defense actions taken
- **Latest Flow Analysis**: Shows the most recent network flow analysis
- **Attack Type Distribution**: Breakdown of detected attack types
- **Recent Threats**: List of recent threats with timestamps
- **Critical Alerts**: High-severity threats requiring immediate attention

### Browser Notifications

The system can send browser notifications for critical threats. When you first start monitoring, your browser will ask for notification permission. Grant permission to receive alerts.

### Stopping Monitoring

Click the "Stop Monitoring" button to pause real-time analysis. You can restart it at any time.

### Resetting Statistics

Click "Reset Stats" to clear all accumulated statistics and start fresh.

## Architecture

### Backend (Flask + SocketIO)

- **WebSocket Server**: Handles real-time connections
- **Monitoring Loop**: Background thread that generates synthetic network flows
- **Model Inference**: Processes flows through the ML model
- **Statistics Tracking**: Maintains real-time statistics

### Frontend (React + Socket.IO Client)

- **LiveMonitor Component**: Main real-time monitoring interface
- **RealTimeStats Component**: Background component that updates dashboard stats
- **WebSocket Client**: Connects to backend for real-time updates

## API Endpoints

### REST Endpoints

- `GET /api/stats` - Get current statistics
- `POST /api/predict` - Upload file for batch prediction
- `GET /api/models` - List available models
- `POST /api/load-model` - Load a specific model

### WebSocket Events

**Client → Server:**
- `start_monitoring` - Start real-time monitoring
- `stop_monitoring` - Stop real-time monitoring
- `reset_stats` - Reset statistics

**Server → Client:**
- `connected` - Connection established
- `realtime_update` - Real-time statistics update
- `critical_alert` - Critical threat detected
- `monitoring_started` - Monitoring started confirmation
- `monitoring_stopped` - Monitoring stopped confirmation
- `stats_reset` - Statistics reset confirmation

## Troubleshooting

### Connection Issues

- Ensure the backend server is running on port 5000
- Check browser console for WebSocket connection errors
- Verify CORS settings if accessing from a different origin

### No Updates Appearing

- Check that monitoring is started (green "Connected" status)
- Verify the model is loaded correctly in the backend
- Check browser console for any errors

### Model Not Found

- Ensure model files are in the `artifacts/` directory
- Check that the model file name matches what the backend expects
- Verify file permissions

## Security Notes

- This is a demonstration system using synthetic data
- For production use, integrate with actual network packet capture tools
- Ensure proper authentication and authorization
- Use HTTPS/WSS in production environments

## Future Enhancements

- Integration with actual packet capture (pcap files)
- Historical data storage and analysis
- Custom alert rules and thresholds
- Multi-user support with role-based access
- Export of real-time monitoring data








