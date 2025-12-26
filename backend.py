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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

ARTIFACTS_DIR = Path("artifacts")
loaded_model = None
loaded_model_name = None
expected_cols = None
classes = None

# Real-time monitoring state
monitoring_active = False
monitoring_thread = None
stats = {
    'total_packets': 0,
    'threats_detected': 0,
    'critical_alerts': 0,
    'defense_actions': 0,
    'attack_types': {},
    'recent_threats': []
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

def generate_synthetic_flow():
    """Generate a synthetic network flow for real-time monitoring"""
    if loaded_model is None or expected_cols is None:
        return None
    
    # Generate random feature values within reasonable ranges
    flow_data = {}
    for col in expected_cols:
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
    
    df = pd.DataFrame([flow_data])
    return df

def process_realtime_flow(df):
    """Process a single flow through the model"""
    if loaded_model is None:
        return None
    
    try:
        X = align_features_auto(df, expected_cols)
        X = clean_numeric(X)
        pred = loaded_model.predict(X)[0]
        
        proba = None
        prob_attack = None
        if hasattr(loaded_model, "predict_proba"):
            proba = loaded_model.predict_proba(X)[0]
            if classes is not None and "benign" in classes:
                benign_idx = list(classes).index("benign")
                prob_attack = float(1.0 - proba[benign_idx])
        
        return {
            'prediction': str(pred),
            'probability': prob_attack,
            'is_attack': str(pred) != "benign",
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error processing flow: {e}")
        return None

def realtime_monitoring_loop():
    """Background thread for real-time monitoring"""
    global monitoring_active, stats
    while monitoring_active:
        try:
            # Generate and process a synthetic flow
            flow_df = generate_synthetic_flow()
            if flow_df is not None:
                result = process_realtime_flow(flow_df)
                if result:
                    stats['total_packets'] += 1
                    
                    if result['is_attack']:
                        stats['threats_detected'] += 1
                        attack_type = result['prediction']
                        stats['attack_types'][attack_type] = stats['attack_types'].get(attack_type, 0) + 1
                        
                        # Add to recent threats (keep last 50)
                        threat_entry = {
                            'id': stats['total_packets'],
                            'type': attack_type,
                            'severity': 'high' if result['probability'] > 0.8 else 'medium',
                            'probability': result['probability'],
                            'timestamp': result['timestamp']
                        }
                        stats['recent_threats'].append(threat_entry)
                        if len(stats['recent_threats']) > 50:
                            stats['recent_threats'].pop(0)
                        
                        # Critical alerts for high probability attacks
                        if result['probability'] > 0.9:
                            stats['critical_alerts'] += 1
                            socketio.emit('critical_alert', threat_entry)
                    
                    # Emit real-time update
                    socketio.emit('realtime_update', {
                        'stats': stats.copy(),
                        'latest_flow': result
                    })
            
            # Random delay between 0.5-2 seconds
            time.sleep(np.random.uniform(0.5, 2.0))
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(1)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'status': 'connected', 'stats': stats})

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start real-time monitoring"""
    global monitoring_active, monitoring_thread
    if not monitoring_active:
        monitoring_active = True
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
        'recent_threats': []
    }
    emit('stats_reset', {'status': 'reset', 'stats': stats})

if __name__ == "__main__":
    # Auto-load the first available model
    files = list_model_files()
    if files:
        try:
            load_model_by_name(files[0])
            print(f"Loaded model: {files[0]}")
        except Exception as e:
            print(f"Auto-load failed: {e}")
    else:
        print("Warning: No model files found in artifacts directory")
    
    # Run with SocketIO
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')