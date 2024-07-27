---
layout: post
title: "Predicting Fraudulent Accident Claims - Part 2"
date: 2024-07-26 22:01:18
categories: Fraud Claim Prediction-concluding
permalink: /posts/Predicting-fraudulent-claims-2 

---


## Introduction
This is the concluding part of the tutorial on predicting auto insurance fraud from the claim dataset using Artificial Intelligence. Readers are encouraged to read the first part of this tutorial [here](https://brightaboh.github.io/posts/Predicting-fraudulent-claims)
 where we defined the training pipeline and the xgboost model used in this inference section. 


## Inference
In this section, we applied the model we developed to new datasets that were unseen by the model during the training phase. This allows us to predict which claims (in the real world) are likely to be fraudulent.  
## Goal
We want to enable auto insurance to be able to determine the legitimacy or otherwise of a claim. To achieve this, the inference pipeline is designed such that predictions are made by either entering an insurance policy number for single processing or uploading an entire Excel file for batch processing. 

## Result
![app](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/predict.png?raw=true)
- The result for the single policy number 10520 shows that the policyholder is a female, aged 40, and this claim could be fraudulent, hence the need for further investigation.

- The results from the  batch processing are saved at the predefined location and show similar predictions:
![batch](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/predict_batch.png?raw=true)



## Application Overview
To enable the usability of the model, we developed a web application (**AI Auto Insurance Fraud Detection System**) in **Streamlit** to handle all the processes and the User Interface(UI). This AI-powered application can be found [here](https://autofrauddetection.streamlit.app)


https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/Screen%20Shot%202024-07-27%20at%202.43.22%20AM.png
![app](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/Screen%20Shot%202024-07-27%20at%202.43.22%20AM.png?raw=true)
The AI Auto Insurance Fraud Detection System is a powerful AI-driven application designed to identify potential fraudulent insurance claims. By leveraging machine learning techniques and a robust preprocessing pipeline, this system provides accurate predictions to help mitigate fraud risks. Users can upload their own claim datasets or rely on a default dataset for fraud detection.


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
To access the [application](https://autofrauddetection.streamlit.app) 

To access the [data and scripts used](https://github.com/BrightABOH/fraud_detection) 


To contact/follow: 
- [LinkedIn](https://www.linkedin.com/in/bright-aboh-b85932ba/) 
- [Email](bright.s.e.aboh@aims-senegal.org)
  


Happy reading and see you in the next one!

