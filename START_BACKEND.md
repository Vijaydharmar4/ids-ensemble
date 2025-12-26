# How to Start the Backend Server

## Quick Start

1. **Open a terminal/command prompt** in the project directory

2. **Start the backend server:**
   ```bash
   python backend.py
   ```

3. **You should see output like:**
   ```
   Loaded model: cicids_multiclass.joblib
   * Running on http://0.0.0.0:5000
   ```

4. **Keep this terminal window open** - the server needs to keep running

5. **In another terminal, start the React frontend:**
   ```bash
   npm start
   ```

## Troubleshooting

### Port 5000 Already in Use
If you get an error that port 5000 is already in use:
- Find and close the process using port 5000
- Or change the port in `backend.py` (line 326) to a different port like 5001

### Model Not Found
If you see "Warning: No model files found":
- Make sure `artifacts/cicids_multiclass.joblib` exists
- Check that you're running from the project root directory

### Dependencies Missing
If you get import errors:
```bash
pip install -r requirements.txt
```

### Windows Firewall
If the frontend can't connect:
- Windows Firewall might be blocking the connection
- Allow Python through Windows Firewall when prompted

## Verification

Once the backend is running, you can test it by visiting:
- http://localhost:5000/api/health

You should see a JSON response with the server status.

## Running in Background (Optional)

### Windows PowerShell:
```powershell
Start-Process python -ArgumentList "backend.py" -WindowStyle Hidden
```

### Or use a terminal multiplexer like `screen` or `tmux` (if installed)



