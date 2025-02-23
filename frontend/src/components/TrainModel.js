import React, { useState } from 'react';
import axios from 'axios';

const TrainModel = () => {
  const [message, setMessage] = useState('');
  const [modelVersion, setModelVersion] = useState('');
  const [accuracy, setAccuracy] = useState(0);
  const [error, setError] = useState('');

  const handleTrainModel = async () => {
    setError('');
    try {
      const response = await axios.post('/train');
      setMessage(response.data.message);
      setModelVersion(response.data.model_version);
      setAccuracy(response.data.accuracy);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while training the model.');
      }
    }
  };

  return (
    <div>
      <h2>Train Model</h2>
      <button onClick={handleTrainModel}>Start Model Training</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {message && (
        <div>
          <p>{message}</p>
          <p>Model Version: {modelVersion}</p>
          <p>Accuracy: {accuracy}</p>
        </div>
      )}
    </div>
  );
};

export default TrainModel;