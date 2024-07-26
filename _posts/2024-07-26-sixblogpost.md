---
layout: post
title: "Predicting Fraudulent Accident Claims   - Part 2"
date: 2024-07-27 22:01:18
categories: Fraud Claim Prediction
permalink: /posts/Predicting-fraudulent-claims 
image: "https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/xgroc.png?raw=true"
---


## Introduction
This is the concluding part of the tutorial on predicting auto insurance fraud from the claim dataset using Artificial Intelligence. Readers are encourage to read the first part of this tutorial
[here](https://brightaboh.github.io/posts/Predicting-fraudulent-claims) where we defined the training pipeline and the xgboost model used in this inference section. 
Traditional methods of fraud detection often rely on manual investigation and rule-based systems, which are time-consuming, labor-intensive, and may not be effective in uncovering sophisticated fraud schemes. However, with advancements in technology, particularly in the field of artificial intelligence and machine learning, insurers now have powerful tools at their disposal to combat insurance fraud more effectively. This blog post is divided into 2 parts, in Part 1, we experiment with different algorithms, and in Part 2 will develop and deploy a web-based App based on the results obtained in Part 1 


## Data loading
We begin by reading the insurance claims data as follows
```python
# Load the data
data = pd.read_excel("claims.xlsx")
```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/xgroc.png?raw=true)
