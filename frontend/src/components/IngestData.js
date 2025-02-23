import React, { useState } from 'react';
import axios from 'axios';

const IngestData = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [recordCount, setRecordCount] = useState(0);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setMessage('');
    setError('');
    setRecordCount(0);

    if (!file) {
      setError('Please select a file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/ingest', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(response.data.message);
      setRecordCount(response.data.record_count);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while uploading the file.');
      }
    }
  };

  return (
    <div>
      <h2>Ingest Data</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="file">Upload File:</label>
          <input type="file" id="file" onChange={handleFileChange} />
        </div>
        <button type="submit">Ingest</button>
      </form>
      {message && (
        <div>
          <p>{message}</p>
          <p>Record Count: {recordCount}</p>
        </div>
      )}
      {error && <div style={{ color: 'red' }}>{error}</div>}
    </div>
  );
};

export default IngestData;