---
layout: post
title: "Predicting Fraudulent Claims from Accidents using Deep Learning - Part 1"
date: 2024-04-25 22:01:18
categories: Fraud Claim Prediction
permalink: /posts/Predicting-fraudulent-claims-with-Deep-learning 
---


## Introduction
In today's fast-paced world, insurance fraud has become a significant concern for insurance companies globally. Fraudulent claims not only lead to financial losses but also tarnish the reputation of insurers and increase premiums for honest policyholders. Among various types of insurance fraud, detecting fraudulent claims stemming from accidents poses a unique challenge due to the intricate nature of accidents and the diverse factors involved.

Traditional methods of fraud detection often rely on manual investigation and rule-based systems, which are time-consuming, labor-intensive, and may not be effective in uncovering sophisticated fraud schemes. However, with advancements in technology, particularly in the field of artificial intelligence and machine learning, insurers now have powerful tools at their disposal to combat insurance fraud more effectively. This blog post is divided into 2 parts, in Part 1, we experimnent with different algorithms and Part 2 will developing and deploying a web based App based on the results obtained in Part 1 


## Data description
The insurance claim dataset contains information related to various insurance claims filed by policyholders. It includes details such as policyholder demographics, accident details, policy information, and claim outcomes. The dataset comprises the following columns:

Month: Month in which the claim was filed.

WeekOfMonth: Week number within the month when the claim was filed.

DayOfWeek: Day of the week when the claim was filed.

Make: Make of the vehicle involved in the accident.

AccidentArea: Area where the accident occurred.

DayOfWeekClaimed: Day of the week when the claim was reported.

MonthClaimed: Month when the claim was reported.

WeekOfMonthClaimed: Week number within the month when the claim was reported.

Sex: Gender of the policyholder.

MaritalStatus: Marital status of the policyholder.

Age: Age of the policyholder.

Fault: Fault attribution for the accident (e.g., policyholder, third party).

PolicyType: Type of insurance policy.

VehicleCategory: Category of the vehicle involved in the accident.

VehiclePrice: Price range of the vehicle.

FraudFound_P: Binary indicator for whether fraud was found in the claim.

PolicyNumber: Unique identifier for the insurance policy.

RepNumber: Representative number associated with the claim.

Deductible: Deductible amount for the claim.

DriverRating: Rating assigned to the driver involved in the accident.

Days_Policy_Accident: Number of days the policy has been active at the time of the accident.

Days_Policy_Claim: Number of days the policy has been active at the time of the claim.

PastNumberOfClaims: Number of claims filed in the past by the policyholder.

AgeOfVehicle: Age of the vehicle involved in the accident.

AgeOfPolicyHolder: Age of the policyholder.

PoliceReportFiled: Binary indicator for whether a police report was filed for the accident.

WitnessPresent: Binary indicator for whether a witness was present at the accident.

AgentType: Type of agent handling the claim.

NumberOfSuppliments: Number of supplementary items included in the claim.

AddressChange_Claim: Binary indicator for whether there was a change of address associated with the claim.

NumberOfCars: Number of cars involved in the accident.

Year: Year in which the claim was filed.

BasePolicy: Base policy associated with the claim.

ClaimSize: Size of the insurance claim.

## Initial EDA
Let's know our data by performing some explorations. We start by looking at the general overview of the data; the dimension of the data, the data types of the various columns, missing values, etc. This gives us an idea of what to expect and the necessary pre-processing.
Assuming you read and save your data in a variable called Data, we can peek into the first few rows by doing this in Python
```
data.head()
```
