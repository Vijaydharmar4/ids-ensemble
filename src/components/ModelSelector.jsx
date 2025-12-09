import React, { useState, useEffect } from 'react';

export function ModelSelector({ onModelLoad }) {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(''); // ADD THIS

  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        // ADD SAFETY CHECK
        const modelList = data.models || [];
        setModels(modelList);
        if (modelList.length > 0) setSelectedModel(modelList[0]);
      })
      .catch(err => {
        setError('Failed to load models');
        console.error('Error loading models:', err);
      });
  }, []);

  const handleLoadModel = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }
    setLoading(true);
    setError(''); // CLEAR ERROR
    try {
      const res = await fetch('/api/load-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: selectedModel })
      });
      // ADD ERROR CHECK
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      onModelLoad(data);
    } catch (err) {
      setError('Error loading model: ' + err.message);
      console.error('Error loading model:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="sidebar">
      <h3>Model Selection</h3>
      {error && <p className="error">{error}</p>}
      {models.length === 0 ? (
        <p className="error">No models found in artifacts/</p>
      ) : (
        <>
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            className="select-input"
          >
            {models.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
          <button 
            onClick={handleLoadModel} 
            disabled={loading}
            className="btn-primary"
          >
            {loading ? '🔄 Loading...' : 'Load Model'}
          </button>
        </>
      )}
    </div>
  );
}