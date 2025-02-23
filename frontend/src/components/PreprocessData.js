import React, { useState } from 'react';
import axios from 'axios';

const PreprocessData = () => {
  const [message, setMessage] = useState('');
  const [recordCount, setRecordCount] = useState(0);
  const [error, setError] = useState('');

  const handlePreprocess = async () => {
    setError('');
    try {
      const response = await axios.post('/preprocess');
      setMessage(response.data.message);
      setRecordCount(response.data.record_count);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while preprocessing the data.');
      }
    }
  };

  return (
    <div>
      <h2>Preprocess Data</h2>
      <button onClick={handlePreprocess}>Start Preprocessing</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {message && (
        <div>
          <p>{message}</p>
          <p>Record Count: {recordCount}</p>
        </div>
      )}
    </div>
  );
};

export default PreprocessData;