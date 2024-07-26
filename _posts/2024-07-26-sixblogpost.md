---
layout: post
title: "Predicting Fraudulent Accident Claims   - Part 2"
date: 2024-07-27 22:01:18
categories: Fraud Claim Prediction
permalink: /posts/Predicting-fraudulent-claims 
image: "https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/xgroc.png?raw=true"
---
https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/stream_app.png?raw=true

## Introduction
This is the concluding part of the tutorial on predicting auto insurance fraud from the claim dataset using Artificial Intelligence. Readers are encouraged to read the first part of this tutorial [here](https://brightaboh.github.io/posts/Predicting-fraudulent-claims)
 where we defined the training pipeline and the xgboost model used in this inference section. 


## Inference
In this section, we applied the model we developed to new datasets that were unseen by the model during the training phase. This allows us to predict which claims (in the real world) are likely to be fraudulent.  
## Goal
We want to enable auto insurance to be able to determine the legitimacy or otherwise of a claim. To achieve this, the inference pipeline is designed such that predictions are made by either entering an insurance policy number for single processing or uploading an entire Excel file for batch processing. The snippet of code below demonstrates how these were achieved

```python
import joblib
import pandas as pd
import numpy as np

#Process the new data as before
class PreprocessingPipeline:

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        self.days = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
            'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        self.vehicle_prices = {
            'less than 20000': 1, '20000 to 29000': 2, '30000 to 39000': 3,
            '40000 to 59000': 4, '60000 to 69000': 5, 'more than 69000': 6,
        }
        self.age_of_vehicle_variants = {
            'new': 0.5, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8.5,
        }
        self.age_variants = {
            '16 to 17': 1, '18 to 20': 2, '21 to 25': 3, '26 to 30': 4,
            '31 to 35': 5, '36 to 40': 6, '41 to 50': 7, '51 to 65': 8, 'over 65': 9,
        }
        self.reverse_maps = {}

    def load_data(self, file_path):
        data = pd.read_excel(file_path)
        return data.copy()
    
    def process_months(self, df, month_columns):
        month_proc = lambda x: self.months.get(x, 0)
        for col in month_columns:
            df[col] = df[col].apply(month_proc)
        return df
    
    def process_days_of_week(self, df, day_columns):
        day_proc = lambda x: self.days.get(x, 0)
        for col in day_columns:
            df[col] = df[col].apply(day_proc)
        return df
    
    def process_vehicle_price(self, df, vehicle_price_column):
        vehicle_price_proc = lambda x: self.vehicle_prices.get(x, 0)
        df[vehicle_price_column] = df[vehicle_price_column].apply(vehicle_price_proc)
        return df
    
    def process_vehicle_age(self, df, vehicle_age_column):
        vehicle_age_proc = lambda x: self.age_of_vehicle_variants.get(x, 0)
        df[vehicle_age_column] = df[vehicle_age_column].apply(vehicle_age_proc)
        return df
    
    def process_policy_holder_age(self, df, age_column):
        age_proc = lambda x: self.age_variants.get(x, 0)
        df[age_column] = df[age_column].apply(age_proc)
        return df
    
    def fill_missing_values(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            df[column] = df[column].fillna(df[column].median())
        return df
    
    def handle_age(self, df, age_column):
        mean_age = df[age_column].mean()
        df[age_column] = df[age_column].apply(lambda x: mean_age if pd.isnull(x) or x < 16 else x)
        return df
    
    def object_to_numerical(self, df):
        for column in df.select_dtypes(include=['object']).columns:
            unique_values = df[column].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            reverse_map = {i: value for value, i in value_map.items()}
            self.reverse_maps[column] = reverse_map
            df[column] = df[column].map(value_map)
        return df
    
    def process_data(self, df):
        df = self.process_months(df, ['Month', 'MonthClaimed'])
        df = self.process_days_of_week(df, ['DayOfWeek', 'DayOfWeekClaimed'])
        df = self.process_vehicle_price(df, 'VehiclePrice')
        df = self.process_vehicle_age(df, 'AgeOfVehicle')
        df = self.process_policy_holder_age(df, 'AgeOfPolicyHolder')
        
        df = self.handle_age(df, 'Age')
        df = self.fill_missing_values(df)
        df = self.object_to_numerical(df)
        
        return df

def get_claim_details_by_policy_number(dataframe, policy_number):
    claim_details = dataframe[dataframe['PolicyNumber'] == policy_number]
    if claim_details.empty:
        raise ValueError(f"No claim found for policy number: {policy_number}")
    return claim_details

# Load the saved pipeline
pipeline = joblib.load('xgboost_model_pipeline.pkl')

def predict_fraud(data, threshold=0.5):
    probabilities = pipeline.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities

def process_csv_for_inference(file_path):
    preprocessing_pipeline = PreprocessingPipeline()
    data = preprocessing_pipeline.load_data(file_path)
    
    # Remove duplicate PolicyNumbers
    data = data.drop_duplicates(subset=['PolicyNumber'])
    
    processed_data = preprocessing_pipeline.process_data(data)
    return processed_data, data, preprocessing_pipeline.reverse_maps

def process_policynumber_for_inference(file_path, policy_number):
    preprocessing_pipeline = PreprocessingPipeline()
    data = preprocessing_pipeline.load_data(file_path)
    
    # Remove duplicate PolicyNumbers
    data = data.drop_duplicates(subset=['PolicyNumber'])
    
    processed_data = preprocessing_pipeline.process_data(data)
    claim_details = processed_data[processed_data['PolicyNumber'] == policy_number]
    if claim_details.empty:
        raise ValueError(f"No claim found for policy number: {policy_number}")
    
    return processed_data, claim_details, data, preprocessing_pipeline.reverse_maps

def predict_from_csv(file_path, threshold=0.5):
    processed_data, original_data, reverse_maps = process_csv_for_inference(file_path)
    predictions, probabilities = predict_fraud(processed_data, threshold)
    
    original_data['FraudProbability'] = probabilities
    original_data['FraudProbability'] = original_data['FraudProbability'].round(2)
    original_data['Prediction'] = predictions
    original_data['Prediction'] = original_data['Prediction'].map({0: 'Legit', 1: 'Possible Fraud'})
    original_data = original_data.drop_duplicates(subset=['PolicyNumber'])
    
    # Reverting numerical values back to original categorical values
    for column, reverse_map in reverse_maps.items():
        original_data[column] = original_data[column].map(reverse_map)
    
    original_data['Age'] = original_data['Age'].astype(int)   # Round age to the nearest number
    
    return original_data[['PolicyNumber', 'Make', 'Sex', 'Age', 'FraudProbability', 'Prediction']]

def predict_from_policy_number(file_path, policy_number, threshold=0.5):
    processed_data, claim_details, original_data, reverse_maps = process_policynumber_for_inference(file_path, policy_number)
    predictions, probabilities = predict_fraud(claim_details, threshold)
    # Add prediction results to the processed claim details
    claim_details['FraudProbability'] = probabilities
    claim_details['Prediction'] = predictions
    claim_details['Prediction'] = claim_details['Prediction'].map({0: 'Legit', 1: 'Possible Fraud'})
    claim_details['Age'] = claim_details['Age'].astype(int)  # Ensure age is a whole number
    
    # Drop duplicates if any
    claim_details = claim_details.drop_duplicates(subset=['PolicyNumber'])
    
    # Reverting numerical values back to original categorical values
    for column, reverse_map in reverse_maps.items():
        claim_details[column] = claim_details[column].map(reverse_map).fillna(claim_details[column])
    
    return claim_details[['PolicyNumber', 'Make', 'Sex', 'Age', 'FraudProbability', 'Prediction']]

def main(file_path=None, policy_number=None, threshold=0.5):
    if file_path:
        if policy_number is not None:
            single_policy_result = predict_from_policy_number(file_path, policy_number, threshold)
            print(single_policy_result)
            return single_policy_result
        
        result = predict_from_csv(file_path, threshold)
        result.to_csv('/Users/brightabohsilasedem/Downloads/claims_predictions_2.csv', index=False)
        print("Predictions saved to 'claims_predictions.csv'")
        return result
    
    raise ValueError("Either file_path or policy_number must be provided.")

```

 The example below shows how the above code can be used:
 - Uploading the claims file for batch processing 
```python
file_path = '/Users/brightabohsilasedem/Downloads/claims.xlsx'
```
- Selecting a single policy number
```python
policy_number = 10520
```
- Batch prediction from the file uploaded
 ```
result = main(file_path=file_path)
```
- Predict from a single policy number

```
single_policy_result = main(file_path=file_path, policy_number=policy_number)
```
## Result
![app](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/predict.png?raw=true)
- The result for the single policy number 10520 shows that the policyholder is a female, aged 40, and this claim could be fraudulent, hence the need for further investigation.

- The results from the  batch processing are saved at the predefined location and show similar predictions:
![batch](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/predict_batch.png?raw=true)



## Application Overview
To enable the usability of the model, we developed a web application (**Intelligent Auto Insurance Fraud Detection System**) in Streamlit to handle all the processes and the User Interface(UI). This AI-powered application can be found [here](https://autofrauddetection.streamlit.app)

![app](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/stream_app.png?raw=true)
The Intelligent Auto Insurance Fraud Detection System is a powerful AI-driven application designed to identify potential fraudulent insurance claims. By leveraging machine learning techniques and a robust preprocessing pipeline, this system provides accurate predictions to help mitigate fraud risks. Users can upload their own claim datasets or rely on a default dataset for fraud detection.


## Features
- AI-Powered Fraud Detection: Utilizes advanced machine learning models to predict the likelihood of fraud in insurance claims.
- Flexible Data Input: Accepts user-uploaded datasets or uses a built-in default dataset.
- Policy Number Lookup: Allows for single claim analysis by entering or selecting a policy number.
- Batch Processing: Processes entire datasets for comprehensive fraud detection.
- Adjustable Threshold: Fine-tune the fraud probability threshold to control sensitivity.
- Interactive Visualization: Presents results in an easily interpretable format with options to download predictions.

## Default Dataset
 For demonstration purposes, the application will automatically use a pre-configured default dataset that mimics a real-world claim dataset. This default dataset contains a representative sample of insurance claims, ensuring that fraud predictions can still be generated even without user-provided data. The default dataset is useful for:

- Testing and Demonstration: Allows users to explore the functionality and capabilities of the application without needing their own data.
- Benchmarking: Provides a baseline for evaluating the performance of the fraud detection system.
- Backup Analysis: Ensures that fraud detection predictions can be made even if no user data is uploaded.

## How It Works
- Default Dataset Handling: If no file is uploaded, the application uses a default dataset stored in the project directory.
- Processing and Prediction: The default dataset is processed using the same preprocessing pipeline as user-uploaded data, and predictions are made using the trained machine learning model.
- Result Display: Results from the default dataset are displayed in the application interface, and users can view and download these predictions.

## Use the Application
- Upload a Claim File: Use the sidebar to upload an Excel file containing insurance claim data.
- Set Fraud Probability Threshold: Adjust the slider to set the threshold for fraud prediction.
- Enter or Select a Policy Number: Enter a policy number manually or select from the dropdown list.
- Submit for Analysis: Click the "Submit" button to process the data and view results.
If no file is uploaded, the system will automatically use the default dataset for analysis.
To predict using the system dataset:
- Submit for Analysis: Click the "Submit" button to process the default data and view the results (batch processing)
- Enter or Select a Policy Number: Enter a policy number manually or select from the dropdown list for single processing.

## Application Interface

- About: Provides information about the application.
- Upload File: Allows users to upload their claim datasets.
- Threshold Slider: Adjusts the sensitivity of fraud detection.
- Policy Number Input: For single claim analysis.
- Results Display: Shows predictions and probabilities for each claim.
- Download Button: Exports the results to a CSV file.

## Miscellaneous 
To access the application: https://autofrauddetection.streamlit.app
To access the data and scripts used: https://github.com/BrightABOH/fraud_detection
To contact/follow: 
- LinkedIn: https://www.linkedin.com/in/bright-aboh-b85932ba/
- Email: bright.s.e.aboh@aims-senegal.org
- website: https://brightaboh.github.io



