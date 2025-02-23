import React, { useState } from 'react';
import axios from 'axios';

const EDAResults = () => {
  const [edaResults, setEdaResults] = useState(null);
  const [error, setError] = useState('');

  const fetchEDAResults = async () => {
    setError('');
    try {
      const response = await axios.get('/eda');
      setEdaResults(response.data);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while fetching the EDA results.');
      }
    }
  };

  return (
    <div>
      <h2>EDA Results</h2>
      <button onClick={fetchEDAResults}>Fetch EDA Results</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {edaResults && (
        <div>
          <h3>Missing Values</h3>
          <ul>
            {Object.entries(edaResults.missing_values).map(([key, value]) => (
              <li key={key}>{key}: {value}</li>
            ))}
          </ul>
          <h3>Correlations</h3>
          <ul>
            {Object.entries(edaResults.correlations).map(([key, value]) => (
              <li key={key}>{key}: {value}</li>
            ))}
          </ul>
          <h3>Visualizations</h3>
          {edaResults.visualizations.map((visualization, index) => (
            <img key={index} src={`data:image/png;base64,${visualization}`} alt={`Visualization ${index + 1}`} />
          ))}
        </div>
      )}
    </div>
  );
};

export default EDAResults;