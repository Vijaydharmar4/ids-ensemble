import React, { useState, useEffect } from 'react';
import './FileUploader.css';

// Use proxy if available (set in package.json), otherwise use full URL
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

export function FileUploader({ onFileProcess }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null);

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const apiUrl = API_BASE_URL ? `${API_BASE_URL}/api/health` : '/api/health';
      const res = await fetch(apiUrl, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        if (data.model_loaded) {
          setBackendStatus('connected');
        } else {
          setBackendStatus('no_model');
          setError('Backend is running but model is not loaded. Please check the artifacts folder.');
        }
      } else {
        setBackendStatus('error');
      }
    } catch (err) {
      setBackendStatus('error');
      console.error('Backend connection check failed:', err);
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    setFileName(file.name);
    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = API_BASE_URL ? `${API_BASE_URL}/api/predict` : '/api/predict';
      
      let res;
      try {
        res = await fetch(apiUrl, {
          method: 'POST',
          body: formData
        });
      } catch (networkError) {
        // Network error (CORS, connection refused, etc.)
        throw new Error(`Cannot connect to backend server. Make sure the Flask backend is running on port 5000. Error: ${networkError.message}`);
      }

      // Get response text first to handle both JSON and text errors
      const responseText = await res.text();
      
      if (!res.ok) {
        let errorMessage = `HTTP ${res.status}: `;
        try {
          const errorData = JSON.parse(responseText);
          errorMessage += errorData.error || errorData.message || 'Server error occurred';
        } catch (e) {
          // If not JSON, use the text response
          errorMessage += responseText || `Server returned status ${res.status}`;
        }
        throw new Error(errorMessage);
      }

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (e) {
        throw new Error(`Invalid response from server: ${responseText.substring(0, 100)}`);
      }
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      if (!data.success && !data.stats) {
        throw new Error('Unexpected response format from server');
      }

      // Ensure attack_counts is in the correct format for AttackBreakdown component
      if (data.attack_counts && typeof data.attack_counts === 'object' && !Array.isArray(data.attack_counts)) {
        data.attack_counts = Object.entries(data.attack_counts).map(([name, count]) => ({
          name,
          count: typeof count === 'number' ? count : parseInt(count) || 0
        }));
      }

      // Set download URL if csv_data is available
      if (data.csv_data) {
        const blob = new Blob([data.csv_data], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        data.download_url = url;
      }

      onFileProcess(data);
      setError(null);
    } catch (err) {
      console.error('Error processing file:', err);
      const errorMessage = err.message || 'Failed to process file. Please check if the backend server is running on port 5000.';
      setError(errorMessage);
      
      // Log full error details for debugging
      console.error('Full error details:', {
        message: err.message,
        stack: err.stack,
        name: err.name
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card file-uploader-card">
      <h2>📤 Upload CICIDS CSV</h2>
      <p className="upload-description">
        Upload a CSV file containing network flow data. The file will be analyzed using the pre-trained ensemble model.
      </p>
      
      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>🤖 Model is processing your file...</p>
          {fileName && <p className="file-name">Processing: {fileName}</p>}
        </div>
      ) : (
        <div className="file-input-container">
          <label htmlFor="file-upload" className="file-upload-label">
            <span className="upload-icon">📁</span>
            <span>Choose CSV File</span>
          </label>
          <input 
            id="file-upload"
            type="file" 
            accept=".csv" 
            onChange={handleFileChange}
            className="file-input"
          />
          {fileName && !loading && (
            <p className="selected-file">Selected: {fileName}</p>
          )}
        </div>
      )}

      {backendStatus === 'error' && (
        <div className="backend-warning">
          <span className="warning-icon">⚠️</span>
          <div>
            <strong>Backend server not detected</strong>
            <p>Make sure the Flask backend is running on port 5000. Run: <code>python backend.py</code></p>
            <button onClick={checkBackendConnection} className="retry-button">
              🔄 Retry Connection
            </button>
          </div>
        </div>
      )}

      {backendStatus === 'no_model' && (
        <div className="backend-warning">
          <span className="warning-icon">⚠️</span>
          <div>
            <strong>Model not loaded</strong>
            <p>Backend is running but the ML model is not loaded. Check that <code>artifacts/cicids_multiclass.joblib</code> exists.</p>
            <button onClick={checkBackendConnection} className="retry-button">
              🔄 Retry Connection
            </button>
          </div>
        </div>
      )}

      {backendStatus === 'connected' && (
        <div className="backend-success">
          <span className="success-icon">✅</span>
          <span>Backend server connected and ready</span>
        </div>
      )}

      <div className="upload-info">
        <h4>Supported Format:</h4>
        <ul>
          <li>CSV files with network flow features</li>
          <li>CICIDS2017 format compatible</li>
          <li>Auto-alignment of columns to model features</li>
        </ul>
      </div>
    </div>
  );
}