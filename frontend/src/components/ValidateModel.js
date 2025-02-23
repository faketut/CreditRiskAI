import React, { useState } from 'react';
import axios from 'axios';

const ValidateModel = () => {
  const [message, setMessage] = useState('');
  const [rocAuc, setRocAuc] = useState(0);
  const [precision, setPrecision] = useState(0);
  const [recall, setRecall] = useState(0);
  const [calibrationDetails, setCalibrationDetails] = useState({});
  const [error, setError] = useState('');

  const handleValidateModel = async () => {
    setError('');
    try {
      const response = await axios.post('/validate');
      setMessage(response.data.message);
      setRocAuc(response.data.roc_auc);
      setPrecision(response.data.precision);
      setRecall(response.data.recall);
      setCalibrationDetails(response.data.calibration_details);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while validating the model.');
      }
    }
  };

  return (
    <div>
      <h2>Validate Model</h2>
      <button onClick={handleValidateModel}>Start Model Validation</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {message && (
        <div>
          <p>{message}</p>
          <p>ROC AUC: {rocAuc}</p>
          <p>Precision: {precision}</p>
          <p>Recall: {recall}</p>
          <h3>Calibration Details</h3>
          <p>Prob True: {calibrationDetails.prob_true && calibrationDetails.prob_true.join(', ')}</p>
          <p>Prob Pred: {calibrationDetails.prob_pred && calibrationDetails.prob_pred.join(', ')}</p>
        </div>
      )}
    </div>
  );
};

export default ValidateModel;