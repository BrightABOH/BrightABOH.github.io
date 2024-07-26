---
layout: post
title: "Predicting Fraudulent Accident Claims   - Part 2"
date: 2024-07-27 22:01:18
categories: Fraud Claim Prediction
permalink: /posts/Predicting-fraudulent-claims 
image: "https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/xgroc.png?raw=true"
---


## Introduction
In today's fast-paced world, insurance fraud has become a significant concern for insurance companies globally. Fraudulent claims not only lead to financial losses but also tarnish the reputation of insurers and increase premiums for honest policyholders. Among various types of insurance fraud, detecting fraudulent claims stemming from accidents poses a unique challenge due to the intricate nature of accidents and the diverse factors involved.

Traditional methods of fraud detection often rely on manual investigation and rule-based systems, which are time-consuming, labor-intensive, and may not be effective in uncovering sophisticated fraud schemes. However, with advancements in technology, particularly in the field of artificial intelligence and machine learning, insurers now have powerful tools at their disposal to combat insurance fraud more effectively. This blog post is divided into 2 parts, in Part 1, we experiment with different algorithms, and in Part 2 will develop and deploy a web-based App based on the results obtained in Part 1 
## Data loading
We begin by reading the insurance claims data as follows
```python
# Load the data
data = pd.read_excel("claims.xlsx")
```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/xgroc.png?raw=true)
