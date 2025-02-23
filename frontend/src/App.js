import React from 'react';
import { IngestData, EDAResults, PreprocessData, SelectFeatures, TrainModel, ValidateModel, Predict } from './components';

const App = () => {
  return (
    <div>
        <h1>Data Ingestion App</h1>
        <IngestData />
        <h1>Data Preprocessing App</h1>
        <PreprocessData />
        <h1>Data Analysis App</h1>
        <EDAResults />
        <h1>Feature Selection App</h1>
        <SelectFeatures />
        <h1>Model Training App</h1>
        <TrainModel />
        <h1>Model Validation App</h1>
        <ValidateModel />
        <h1>Model Prediction App</h1>
        <Predict />
    </div>
  );
};

export default App;