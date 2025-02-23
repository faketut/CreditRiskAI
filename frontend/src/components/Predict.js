import React, { useState } from 'react';
import axios from 'axios';

const Predict = () => {
  const [customerData, setCustomerData] = useState({
    // Initialize with the fields expected by the API
    customer_id: '',
    month: '',
    name: '',
    age: '',
    ssn: '',
    occupation: '',
    annual_income: '',
    monthly_inhand_salary: '',
    num_bank_accounts: '',
    num_credit_card: '',
    interest_rate: '',
    num_of_loan: '',
    type_of_loan: '',
    delay_from_due_date: '',
    num_of_delayed_payment: '',
    changed_credit_limit: '',
    num_credit_inquiries: '',
    credit_mix: '',
    outstanding_debt: '',
    credit_utilization_ratio: '',
    credit_history_age: '',
    payment_of_min_amount: '',
    total_emi_per_month: '',
    amount_invested_monthly: '',
    payment_behaviour: '',
    monthly_balance: '',
    credit_score: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [modelVersion, setModelVersion] = useState('');
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setCustomerData({
      ...customerData,
      [e.target.name]: e.target.value
    });
  };

  const handlePredict = async () => {
    setError('');
    try {
      const response = await axios.post('/predict', { customer_data: customerData });
      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
      setModelVersion(response.data.model_version);
    } catch (error) {
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while making the prediction.');
      }
    }
  };

  return (
    <div>
      <h2>Predict Credit Risk</h2>
      <form>
        {Object.keys(customerData).map((key) => (
          <div key={key}>
            <label htmlFor={key}>{key.replace(/_/g, ' ')}:</label>
            <input
              type="text"
              id={key}
              name={key}
              value={customerData[key]}
              onChange={handleChange}
            />
          </div>
        ))}
      </form>
      <button onClick={handlePredict}>Predict</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {prediction !== null && (
        <div>
          <p>Prediction: {prediction}</p>
          <p>Confidence: {confidence}</p>
          <p>Model Version: {modelVersion}</p>
        </div>
      )}
    </div>
  );
};

export default Predict;