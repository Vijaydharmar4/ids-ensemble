# Backend Server Troubleshooting

## Issue: HTTP 404 - Cannot POST /api/predict

This error means something is running on port 5000, but it's NOT the correct Flask backend.

## Solution Steps:

### 1. Stop Everything Running on Port 5000

**Find what's running:**
```powershell
netstat -ano | findstr :5000
```

**Kill the process (replace PID with the number from above):**
```powershell
taskkill /PID <PID> /F
```

### 2. Make Sure You're Running the Correct File

**WRONG:** `python app.py` (This is Streamlit, not Flask API)
**CORRECT:** `python backend.py` (This is the Flask API server)

### 3. Start the Backend Correctly

Open a **NEW** terminal window and run:

```bash
cd "C:\Users\rohit\OneDrive\Documents\Major project\ids-ensemble"
python backend.py
```

**You should see:**
```
Loaded model: cicids_multiclass.joblib
 * Running on http://0.0.0.0:5000
```

### 4. Verify the Backend is Running

Open a browser and go to: http://localhost:5000/

You should see JSON with available endpoints.

Or test with:
```powershell
curl http://localhost:5000/api/health
```

### 5. Common Issues

**If you see "Module not found":**
```bash
pip install flask flask-cors flask-socketio eventlet pandas numpy scikit-learn joblib
```

**If port 5000 is busy:**
- Close all Python processes
- Or change port in `backend.py` line 326 to `port=5001`

**If model not found:**
- Check that `artifacts/cicids_multiclass.joblib` exists
- Make sure you're in the project root directory

### Quick Start Scripts

I've created helper scripts:
- **Windows:** Double-click `start_backend.bat`
- **PowerShell:** Run `.\start_backend.ps1`



