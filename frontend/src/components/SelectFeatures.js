import React, { useState } from 'react';
import axios from 'axios';

const SelectFeatures = () => {
  const [message, setMessage] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [error, setError] = useState('');

  const handleSelectFeatures = async () => {
    setError('');
    try {
      const response = await axios.post('/select_features');
      setMessage(response.data.message);
      setSelectedFeatures(response.data.selected_features);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while selecting features.');
      }
    }
  };

  return (
    <div>
      <h2>Select Features</h2>
      <button onClick={handleSelectFeatures}>Start Feature Selection</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {message && (
        <div>
          <p>{message}</p>
          <h3>Selected Features</h3>
          <ul>
            {selectedFeatures.map((feature, index) => (
              <li key={index}>{feature}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SelectFeatures;