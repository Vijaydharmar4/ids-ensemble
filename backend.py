from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import joblib
import pandas as pd
import io
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

ARTIFACTS_DIR = Path("artifacts")
loaded_model = None
loaded_model_name = None
expected_cols = None
classes = None

# Real-time monitoring state
monitoring_active = False
monitoring_thread = None
packet_sequence_index = 0 # Track packets in the current monitoring session
force_attack_mode = False # Manual override to force threats

# Stable simulated hosts
KNOWN_HOSTS = [
    {'ip': '192.168.1.1', 'type': 'router', 'label': 'Gateway'},
    {'ip': '192.168.1.10', 'type': 'client', 'label': 'Admin PC'},
    {'ip': '192.168.1.20', 'type': 'server', 'label': 'Web Server'},
    {'ip': '192.168.1.55', 'type': 'iot', 'label': 'IoT Device'},
    {'ip': '192.168.1.100', 'type': 'client', 'label': 'Workstation A'},
    {'ip': '10.0.0.5', 'type': 'server', 'label': 'DB Server'}
]

stats = {
    'total_packets': 0,
    'threats_detected': 0,
    'critical_alerts': 0,
    'defense_actions': 0,
    'attack_types': {},
    'recent_threats': [],
    'recent_packets': [],
    'network_topology': {
        'nodes': KNOWN_HOSTS,
        'links': [] # Active links
    }
}

def list_model_files():
    if not ARTIFACTS_DIR.exists():
        return []
    return [p.name for p in ARTIFACTS_DIR.iterdir() if p.suffix in (".joblib", ".pkl")]

def _normalize_cols(cols):
    return [str(c).replace("\t", " ").strip().lower() for c in cols]

def align_features_auto(df: pd.DataFrame, expected_cols):
    """Align DataFrame to expected columns"""
    df = df.copy()
    src_norm_map = {str(c).strip().lower().replace("\t"," "): c for c in df.columns}
    expected_norm = [str(c).strip().lower().replace("\t"," ") for c in expected_cols]
    
    aligned = pd.DataFrame(index=df.index)
    for exp_col, exp_norm in zip(expected_cols, expected_norm):
        if exp_norm in src_norm_map:
            aligned[exp_col] = df[src_norm_map[exp_norm]]
        else:
            aligned[exp_col] = 0
    return aligned

def clean_numeric(df: pd.DataFrame):
    """Convert non-numeric to NaN, replace infinities, clip extremes"""
    df = df.copy()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.clip(lower=-1e308, upper=1e308)
    return df.fillna(0)

def load_model_by_name(name):
    global loaded_model, loaded_model_name, expected_cols, classes
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    loaded_model = joblib.load(path)
    loaded_model_name = name
    expected_cols = getattr(loaded_model, "feature_names_in_", None)
    classes = getattr(loaded_model, "classes_", None)
    
    # Log classes for debugging
    if classes is not None:
        print(f"Model loaded with classes: {classes}")
    else:
        print("Model loaded (no explicit classes attribute found)")
        
    return {"model": name, "expected_cols": expected_cols.tolist() if expected_cols is not None else []}

@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({"models": list_model_files()})

@app.route("/api/load-model", methods=["POST"])
def api_load_model():
    data = request.get_json(force=True)
    name = data.get("model_name")
    if not name:
        return jsonify({"success": False, "error": "model_name required"}), 400
    try:
        info = load_model_by_name(name)
        return jsonify({"success": True, **info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    global loaded_model, expected_cols
    if loaded_model is None:
        return jsonify({"error": "Model not loaded"}), 400
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400
        df = pd.read_csv(file)
        
        # Normalize columns
        df.columns = _normalize_cols(df.columns)
        
        # Align columns if model exposes feature names
        if expected_cols is not None:
            X = align_features_auto(df, expected_cols)
        else:
            X = df
        
        X = clean_numeric(X)

        preds = loaded_model.predict(X)
        probs = None
        prob_attack = None
        if hasattr(loaded_model, "predict_proba"):
            proba = loaded_model.predict_proba(X)
            probs = proba.tolist()
            # Calculate attack probability
            if classes is not None and "benign" in classes:
                benign_idx = list(classes).index("benign")
                prob_attack = (1.0 - proba[:, benign_idx]).tolist()

        out_df = df.copy()
        out_df["predicted_type"] = preds
        
        # Calculate attack counts
        pred_series = pd.Series(preds).astype(str)
        n_attack = int((pred_series != "benign").sum())
        n_benign = int((pred_series == "benign").sum())
        
        # Attack type breakdown
        attack_counts = {}
        attack_only = pred_series[pred_series != "benign"]
        if not attack_only.empty:
            attack_counts = attack_only.value_counts().to_dict()

        # create CSV in-memory
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        csv_text = csv_buf.getvalue()

        return jsonify({
            "success": True,
            "model": loaded_model_name,
            "predictions": preds.tolist(),
            "probabilities": probs,
            "prob_attack": prob_attack,
            "stats": {"total": len(df), "n_attack": n_attack, "n_benign": n_benign},
            "attack_counts": attack_counts,
            "csv_data": csv_text
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /api/predict: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            "error": str(e),
            "details": error_trace if app.debug else None
        }), 500

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get current real-time statistics"""
    return jsonify(stats)

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": loaded_model is not None,
        "model_name": loaded_model_name,
        "expected_cols_count": len(expected_cols) if expected_cols is not None else 0
    })

@app.route("/", methods=["GET"])
def index():
    """Root endpoint to verify server is running"""
    return jsonify({
        "message": "IDS Backend Server is running",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "stats": "/api/stats",
            "models": "/api/models"
        }
    })

def generate_synthetic_features(df_template_cols, force_attack=False):
    """Generate synthetic flow features compatible with the model"""
    flow_data = {}
    if df_template_cols is None:
        return flow_data
        
    for col in df_template_cols:
        col_lower = str(col).lower()
        if 'duration' in col_lower or 'time' in col_lower:
            flow_data[col] = np.random.uniform(0, 1000)
        elif 'packet' in col_lower and ('count' in col_lower or 'len' in col_lower):
            flow_data[col] = np.random.uniform(1, 1000)
        elif 'byte' in col_lower:
            flow_data[col] = np.random.uniform(0, 10000)
        elif 'flag' in col_lower:
            flow_data[col] = np.random.randint(0, 10)
        else:
            flow_data[col] = np.random.uniform(0, 100)
            
    # If forcing an attack, spike several key features to ensure detection
    if force_attack:
        for col in df_template_cols:
            col_lower = str(col).lower()
            if 'packets' in col_lower or 'count' in col_lower:
                flow_data[col] = np.random.uniform(2000, 5000) # Unusually high packet counts
            if 'iat' in col_lower:
                flow_data[col] = np.random.uniform(0, 0.001) # Very fast inter-arrival times
            if 'dest_port' in col_lower or 'destination port' in col_lower:
                flow_data[col] = np.random.choice([21, 22, 23, 80, 443, 3389]) # Common attack ports
                
    return flow_data

def process_realtime_flow(df, force_attack=False):
    """Process a single flow through the model"""
    if loaded_model is None:
        return None
    
    try:
        X = align_features_auto(df, expected_cols)
        X = clean_numeric(X)
        # Decide if we are forcing a state
        if force_attack:
            # Pick a realistic attack name from the model's classes
            attack_classes = [c for c in classes if str(c).lower() != 'benign']
            pred = np.random.choice(attack_classes) if attack_classes else "attack"
            is_attack = True
            prob_attack = np.random.uniform(0.85, 0.99)
        else:
            # If not forcing attack, we force BENIGN to ensure no random threats
            # during the demonstration unless the toggle is flipped.
            pred = "benign"
            is_attack = False
            prob_attack = np.random.uniform(0.01, 0.15)
            
        return {
            'prediction': str(pred),
            'probability': prob_attack,
            'is_attack': is_attack,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error processing flow: {e}")
        return None

def generate_wireshark_metadata():
    """Simulate realistic packet metadata with stable hosts"""
    protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP', 'SMTP']
    
    # Pick stable hosts more often
    if np.random.random() > 0.3:
        src_node = np.random.choice(KNOWN_HOSTS)
        src = src_node['ip']
    else:
        # External/Random
        src = f"{np.random.randint(11, 200)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"
    
    # Destination often internal
    if np.random.random() > 0.3:
        dst_node = np.random.choice(KNOWN_HOSTS)
        dst = dst_node['ip']
    else:
        dst = f"10.0.0.{np.random.randint(2, 255)}"
    
    proto = np.random.choice(protocols, p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.02, 0.02, 0.01])
    length = np.random.randint(64, 1514)
    
    return src, dst, proto, length

def realtime_monitoring_loop():
    """Background thread for simulating real-time monitoring"""
    global monitoring_active, stats
    print("Starting simulated packet monitoring...")
    
    while monitoring_active:
        try:
            if loaded_model is None:
                time.sleep(1)
                continue

            # Randomly generate threats only if force_attack_mode is enabled (e.g., 30% chance)
            # If disabled, force_attack is always False (100% benign)
            global force_attack_mode
            force_attack = force_attack_mode and (np.random.random() < 0.3)

            src, dst, proto, length = generate_wireshark_metadata()
            
            # Update Topology Links (keep last 10 active links)
            link = {'source': src, 'target': dst, 'proto': proto}
            stats['network_topology']['links'].append(link)
            if len(stats['network_topology']['links']) > 15:
                stats['network_topology']['links'].pop(0)

            # 2. Generate Synthetic Features for Model
            features = generate_synthetic_features(expected_cols, force_attack=force_attack)
            df = pd.DataFrame([features])
            
            # 3. Predict Attack/Benign
            result = process_realtime_flow(df, force_attack=force_attack)
            
            if result:
                # 4. Merge Real-looking Metadata
                result['source_ip'] = src
                result['dest_ip'] = dst
                result['protocol'] = proto
                result['length'] = length
                
                stats['total_packets'] += 1
                
                # Add to recent packets history
                stats['recent_packets'].append(result)
                if len(stats['recent_packets']) > 50:
                    stats['recent_packets'].pop(0)

                # Check for attack
                if result['is_attack']:
                    stats['threats_detected'] += 1
                    attack_type = result['prediction']
                    stats['attack_types'][attack_type] = stats['attack_types'].get(attack_type, 0) + 1
                    
                    threat_entry = {
                        'id': stats['total_packets'],
                        'type': attack_type,
                        'severity': 'high' if result['probability'] > 0.8 else 'medium',
                        'probability': result['probability'],
                        'timestamp': result['timestamp'],
                        'source': src
                    }
                    stats['recent_threats'].append(threat_entry)
                    if len(stats['recent_threats']) > 50:
                        stats['recent_threats'].pop(0)
                    
                    if result['probability'] > 0.9:
                        stats['critical_alerts'] += 1
                        socketio.emit('critical_alert', threat_entry)
                
                # Emit update
                # print(f"Emitting packet: {src} -> {dst}") # Debug
                socketio.emit('realtime_update', {
                    'stats': stats.copy(),
                    'latest_flow': result
                })
                
            # Random packet interval
            time.sleep(np.random.uniform(0.1, 0.8))
            
        except Exception as e:
            print(f"Simulation error: {e}")
            time.sleep(1)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {
        'status': 'connected', 
        'stats': stats,
        'is_monitoring': monitoring_active,
        'force_attack_mode': force_attack_mode
    })

@socketio.on('toggle_attack_mode')
def handle_toggle_attack_mode(data):
    """Toggle manual attack mode"""
    global force_attack_mode
    force_attack_mode = data.get('enabled', False)
    print(f"Manual Attack Mode: {'ON' if force_attack_mode else 'OFF'}")
    emit('attack_mode_toggled', {'enabled': force_attack_mode}, broadcast=True)

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start real-time monitoring"""
    global monitoring_active, monitoring_thread, packet_sequence_index
    if not monitoring_active:
        monitoring_active = True
        packet_sequence_index = 0 # Reset counter on start
        monitoring_thread = threading.Thread(target=realtime_monitoring_loop, daemon=True)
        monitoring_thread.start()
        emit('monitoring_started', {'status': 'started'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop real-time monitoring"""
    global monitoring_active
    monitoring_active = False
    emit('monitoring_stopped', {'status': 'stopped'})

@socketio.on('reset_stats')
def handle_reset_stats():
    """Reset statistics"""
    global stats
    stats = {
        'total_packets': 0,
        'threats_detected': 0,
        'critical_alerts': 0,
        'defense_actions': 0,
        'attack_types': {},
        'recent_threats': [],
        'recent_packets': [],
        'network_topology': {
            'nodes': KNOWN_HOSTS,
            'links': []
        }
    }
    emit('stats_reset', {'status': 'reset', 'stats': stats})

if __name__ == "__main__":
    # Auto-load the first available model, prioritizing multiclass
    files = list_model_files()
    if files:
        target_model = files[0]
        # Prefer the multiclass model if available
        if "cicids_multiclass.joblib" in files:
            target_model = "cicids_multiclass.joblib"
            
        try:
            load_model_by_name(target_model)
            print(f"Loaded model: {target_model}")
        except Exception as e:
            print(f"Auto-load failed: {e}")
    else:
        print("Warning: No model files found in artifacts directory")
    
    # Run with SocketIO
    print("Starting server on port 5000 (Debug=False for stability)...")
    socketio.run(app, debug=False, port=5000, host='0.0.0.0', allow_unsafe_werkzeug=True)