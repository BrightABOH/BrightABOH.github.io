---
layout: post
title: "Predicting Fraudulent Accident Claims   - Part 1"
date: 2024-07-16 22:01:18
categories: Fraud Claim Prediction
permalink: /posts/Predicting-fraudulent-claims 
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




## Data intro
Let's know our data by performing some explorations. We start by looking at the general overview of the data; the dimension of the data, the data types of the various columns, missing values, etc. This gives us an idea of what to expect and the necessary pre-processing, we can peek into the first few rows by doing this in Python
```python
Data.head()
```
This operation gives us the first 10 rows of the data(Mostly there is little to see at this point). Next, the ``` data.info() ``` gives us an overview of the data as seen below
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/datainfo.png?raw=true)

Looking at the result of the operation ```data.info```,  we can tell that there are 11565 data points with 34 columns. Out of these columns, 3 of them are of float type, 7 are integers and the rest 24 are objects. Furthermore, we can observe some missing values in some columns, specifically, there are  missing values in Age and DriverRating. How many missing values are in these columns? we can find them out by this operation ``` data.isnull().sum()``` 

![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/missin.png?raw=true)
As we can see there are 5 missing values in Age and 6 missing values in DriverRating. It is a good thing that there are only a few missing values in the dataset). However, irrespective of this small number of missing values, we cannot proceed with our modeling without addressing this issue. We have to decide whether we are going to keep these data points or drop them(we will come back to this later). We examine the distribution of the target variable to understand the class balance in our task. Since our goal is to classify a claim as either legitimate or fraudulent, the target variable is FraudFound_P. FraudFound_P is a binary indicator for whether fraud was found in the claim or not (i.e 0 if there is no fraud and 1 if there is fraud). This ```(data['FraudFound_P'] == 1).sum()``` gives us the number of fraudulent claims whiles ````(data['FraudFound_P'] == 0).sum() ```` gives us the number of legitimate claims. The results of the two operations show there are 10880 legitimate claims and only 685 fraudulent claims (not too surprising). In order words, only ~6\% of our datasets are fraudulent If you think about it, most of the claims filed will naturally be legitimate only a few will be fraudulent, same ideology can be extended to receiving spam emails. Most of your emails will be genuine and only  a few will be in your spam folder.  This phenomenon introduces us to what we call imbalanced data. Generally, if you have a split ratio of 90:10 in the variable of a binary classification, this is pretty obvious that the  dataset is imbalanced

As you've already seen, a data imbalance is a classification problem where there is an unequal distribution of classes within the dataset. 

As you will later see in this tutorial, we must handle the case of imbalances in the data. In the next section, we prepare our data for modeling including fixing the missing values, handling the class imbalance, converting column types into the appropriate types for the chosen algorithms etc

### Data preparation and pre-processing

The first thing we want to address is that of the missing values. I have decided to keep these records, hence I need to choose an appropriate method to fill in these missing values (in the Age and DriverRating columns). I can impute these missing values using the mean, mode, or median values of the variable. I can also forward-fill or backward-fill with the last known or next value in that column. Additionally, the functions below also process the object type data(e.g assigning numerical values to months of the year) by assigning numerical values to the non-numerical and categorical data.


```python
#Preprocessing functions
import pandas as pd

# Load your dataset
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data.copy()

# Function to process month columns
def process_months(df, month_columns):
    months = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    month_proc = lambda x: months.get(x, 0)
    
    for col in month_columns:
        df[col] = df[col].apply(month_proc)
    return df

# Function to process day of week columns
def process_days_of_week(df, day_columns):
    days = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
        'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }
    day_proc = lambda x: days.get(x, 0)
    
    for col in day_columns:
        df[col] = df[col].apply(day_proc)
    return df

# Function to process vehicle price
def process_vehicle_price(df, vehicle_price_column):
    vehicle_prices = {
        'less than 20000': 1, '20000 to 29000': 2, '30000 to 39000': 3,
        '40000 to 59000': 4, '60000 to 69000': 5, 'more than 69000': 6,
    }
    vehicle_price_proc = lambda x: vehicle_prices.get(x, 0)
    
    df[vehicle_price_column] = df[vehicle_price_column].apply(vehicle_price_proc)
    return df

# Function to process vehicle age
def process_vehicle_age(df, vehicle_age_column):
    AgeOfVehicle_variants = {
        'new': 0.5, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8.5,
    }
    vehicle_age_proc = lambda x: AgeOfVehicle_variants[x]
    
    df[vehicle_age_column] = df[vehicle_age_column].apply(vehicle_age_proc)
    return df

# Function to process policy holder age
def process_policy_holder_age(df, age_column):
    age_variants = {
        '16 to 17': 1, '18 to 20': 2, '21 to 25': 3, '26 to 30': 4,
        '31 to 35': 5, '36 to 40': 6, '41 to 50': 7, '51 to 65': 8, 'over 65': 9,
    }
    age_proc = lambda x: age_variants[x]
    
    df[age_column] = df[age_column].apply(age_proc)
    return df

# Function to fill missing values
def fill_missing_values(df, columns_with_default_values):
    for column, default_value in columns_with_default_values.items():
        df[column] = df[column].fillna(default_value)
    return df

# Main processing function
def process_data(file_path):
    df = load_data(file_path)
    
    df = process_months(df, ['Month', 'MonthClaimed'])
    df = process_days_of_week(df, ['DayOfWeek', 'DayOfWeekClaimed'])
    df = process_vehicle_price(df, 'VehiclePrice')
    df = process_vehicle_age(df, 'AgeOfVehicle')
    df = process_policy_holder_age(df, 'AgeOfPolicyHolder')
    
    df = fill_missing_values(df, {'Age': df['Age'].mean(), 'DriverRating': df['DriverRating'].mean()})
    
    return df

# File path to the dataset
file_path = "claims.xlsx"

# Process the data
df_processed = process_data(file_path)
```
Calling the main processing function convert_to_numerical on our data like this convert_to_numerical(data) will ensure that all non-numerical columns have been assigned their numerical representation.  Doing this ```data["Sex"]``` will now give us 1,0,0 as desired. 

## Initial Data Exploratory Analysis(DEA)

Next, we look at how the distribution of some of the columns and their relationship with the target variable(FraudFound_P).
First of all, we look at the age distribution within the dataset with the following snippet;
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data

# Set the style
sns.set(style="whitegrid")

# Create figure and axis
plt.figure(figsize=(14, 8))

# Plot histogram
n, bins, patches = plt.hist(df_processed['Age'], bins=40, edgecolor='black', alpha=0.7)

# Add colors
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(i / len(patches)))

# Add title and labels
plt.title('Age Distribution', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/age_distribution.png?raw=true)
From the graph above, see close to 300 claims with 0 years which doesn't make sense, as babies don't drive. We replace these records with the mean age within the dataset by the following snippet of code;

```
##Drop age 0 as babies dont drive
# Calculate the median age of the drivers
median_age = df_processed[df_processed['Age'] != 0]['Age'].median()

# Replace the age values that are 0 with the median age
df_processed['Age'] = df_processed['Age'].replace(0, median_age)
```
 
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/age_distribution2.png?raw=true)
The result shows 0 years replaced with the mean of the ages. 

Secondly, we can have an insight into the age distribution with fraudulent claims. The snippet code below shows which ages are more likely to have a fraudulent claim. To achieve this, we plot the Kernel Density Estimation graph as follows;
```
# Set the style
sns.set(style="whitegrid")

# Create FacetGrid
g = sns.FacetGrid(df_processed, hue='FraudFound_P', height=7, aspect=2, palette='viridis')

# Map the KDE plot to the grid
g.map(sns.kdeplot, 'Age', shade=True)

# Add a title
plt.title('Age Distribution for Fraud and No Fraud Claims', fontsize=20)

# Add legend
g.add_legend(title="Fraud Found")

# Customize ticks and labels
plt.xlabel('Age', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


ax = plt.gca()
for i, line in enumerate(ax.get_lines()):
    x_data = line_get_xdata()
    y_data = line_get_ydata()
    peak_index = np.argmax(ydata)
    peak_x = x_data[peak_index]
    peak_y = y_data[peak_index]
    ax.fill_between(x_data, 0, y_data, where = (x_data<=peak_x), colors = 'red', alpha = 0.3)
    

# Show the plot
plt.show()

```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/kde.png?raw=true)

The results show that most of the fraudulent claims are around the 30-40 year brackets. This makes sense since the age of most of the drivers in this dataset are in this age group.
Another interesting insight we could drive is to investigate the distribution of fraudulent claims across the sex of the drivers. 
```
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame and 'Sex' and 'FraudFound_P' are columns in it
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', hue='FraudFound_P', data=df_processed, palette='viridis')

# Add title and labels
plt.title('Sex Distribution of Fraud Found', fontsize=20)
plt.xlabel('Sex', fontsize=15)
plt.ylabel('Count', fontsize=15)

# Customize ticks and labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add legend
plt.legend(title='Fraud Found')

# Show the plot
plt.show()

```
What we observe is that there are a lot more fraudulent claimers who are males. Without looking at the sex distribution within the dataset as a whole, one may be tempted to say that male drivers are more likely to commit fraudulent claims than their female counterparts. If we look at the sex distribution, we see that male drivers outnumber female drivers hence the possibility to find  more fraudulent claims in the male category.
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/sex_fraud.png?raw=true)

What about the car makes that are frequently involved in fraudulent claims? Can that be established? The snippet of code below provides an insight of the car makes which are frequently involved in fraudulent claims.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

fraud_rate_make = df_processed.groupby('Make').agg({
    "FraudFound_P": "mean", 
    "PolicyNumber": 'count'
})
fraud_rate_make.columns = ['FraudRate', 'Count']
fraud_rate_make = fraud_rate_make.apply(lambda x: round(x, 3))
fraud_rate_make = fraud_rate_make.sort_values(by='FraudRate', ascending=False)

# Reset index to turn 'Make' into a column for easier plotting
fraud_rate_make.reset_index(inplace=True)

# Plotting
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='FraudRate', y='Make', data=fraud_rate_make, palette='viridis')

# Add counts to the bars
for index, value in enumerate(fraud_rate_make['FraudRate']):
    barplot.text(value, index, f'{value:.2%}', color='black', ha="left", va="center")

# Add title and labels
plt.title('Fraud Rate by Make', fontsize=20)
plt.xlabel('Fraud Rate', fontsize=15)
plt.ylabel('Make', fontsize=15)

# Show the plot
plt.show()
```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/make.png?raw=true)
The result shows that 33.30% of all the fraudulent claims recorded involved a Mercedes car, followed by Accura at 12.70%.

Another aspect to consider is analyzing the WitnessPresent column to gain insights into how the presence of a witness contributes to fraud cases. To achieve this; we use this block of codes
```python
import pandas as pd
import matplotlib.pyplot as plt



# Filter data where FraudFound_P is 1
df_fraud = df_processed[df_processed['FraudFound_P'] == 1]

# Group data by 'WitnessPresent' and count the number of policies
count_rep_wit = df_fraud.groupby('WitnessPresent').agg({
    "PolicyNumber": 'count'
}).reset_index()

# Rename columns
count_rep_wit.columns = ['WitnessPresent', 'Count']

# Plot the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x='WitnessPresent', y='Count', data=count_rep_wit, palette='viridis')

# Add labels to each bar
for index, row in count_rep_wit.iterrows():
    plt.text(row.name, row['Count'], row['Count'], color='black', ha="center", va="bottom")

# Add title and labels
plt.title('Count of Policies by Witness Presence in Fraud Cases', fontsize=20)
plt.xlabel('Witness Present', fontsize=15)
plt.ylabel('Count of Policies', fontsize=15)

# Customize ticks and labels
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()

```


We observe an interesting trend: out of 685 fraudulent claims, 683 (representing 99.7%) occurred at locations where no witnesses were present. This suggests that, in the absence of witnesses at the accident scene, a claim is highly likely to be fraudulent.


## Model training 

We start with a simple logistic model, where FraudFound_P is our target variable, and the rest of the columns as our predictor variables. Note that if we so desire, we can start with a simple confusion matrix to understand the relationship among the predictor variables and possible dimension reduction to include only needed features. However, this approach is not so necessary in our case as we will be employing deep learning for feature engineering, and we need to understand how each of the features will contribute to our final model
We start with a simple logistic regression(with the class imbalance), as  below;
```
#Logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Define features and target variable
X = data.drop(columns=['FraudFound_P'])
y = data['FraudFound_P']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model with class weights
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

```
The model has an accuracy of 94.07%. Not to be too happy, we inspect the class performance. We compute the confusion matrix to understand the performance of the model beyond the high accuracy
```
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['Legit', 'Fraud']
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, annot_kws={"size": 14}, 
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(ticks=np.arange(2) + 0.5, labels=class_names)
plt.yticks(ticks=np.arange(2) + 0.5, labels=class_names)
plt.show()
```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/confusion.png?raw=true)
Observing the results of the confusion matrix from our model, it is clear that our model is doing well in predicting legitimate claims(99.9%) of the time. However this is not our task, our goal is to predict fraudulent claims which our model is so horrible at predicting (0.74%). The model is skewed toward the majority class, therefore despite the 94% accuracy recorded, our model has failed to solve the intended task. To address this is to address the imbalance problem in the dataset, to do this we will experiment with oversamplling the minority class, and adding class weights to the different classes accordingly. 
We start with SMOTE which creates synthetic samples for the minority class. This way, we will increase the number of minority classes thereby solving the class imbalance issue
```
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

```
The model has an accuracy of 61%. As with the first model, we inspect the class performance with the confusion matrix to understand the performance of the model beyond the accuracy 

```
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['Legit', 'Fraud']
# Plot confusion matrix
plt.figure(figsize=(8, 6))
# Calculate class percentages
class_percentages = cm / cm.sum(axis=1)[:, np.newaxis]
# Plot confusion matrix with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Class Counts and Percentages)')
plt.xticks(ticks=np.arange(2) + 0.5, labels=class_names)
plt.yticks(ticks=np.arange(2) + 0.5, labels=class_names)

# Add text annotations for class percentages
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # Compute percentage if count is not zero
        if cm[i, j] != 0:
            percentage = class_percentages[i, j]
            plt.text(j + 0.5, i + 0.2, f'{percentage:.2%}', 
                     horizontalalignment='center', verticalalignment='center', color='green')

plt.show()

```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/confusion2.png?raw=true)
The first thing we observe here is the drop in the overall accuracy from 94% to 61% just after solving the imbalance problem. Does this help solve our problem? Yes!, from not being able to predict any fraudulent claims in the previous model, we can predict 65 fraudulent claims as true fraudulent(True negatives). That notwithstanding, our model is still predicting some 71 fraudulent claims as legit (False negative). Our objective hereafter is to increase the number of fraudulent claims that are indeed predicted as fraudulent by reducing the number of false negative claims(reduce the 71 as low as possible) even if it means increasing the number of legit claims as fraudulent (False positive; 820 in our case). Think about it, it is better to predict legit claims as fraudulent which may turn out not to be fraudulent upon investigation rather than predicting a fraudulent claim as legit which could cause us to lose millions.  
Now that we can deal with the class imbalance, we experiment with other models to reduce the false negatives. 

## Improving performance 
We experiment with a different approach to handle class imbalance and observe the performance of the mode. We introduce the notion of class weights. In this approach, we assign weights to the Legit and Fraud classes; by assigning higher weights to the minority class(Fraud) and lower weights to the majority class(Legit), the model is trained to pay more attention to the minority class samples during the optimization process. In determining how to assign these weights, we use sklearn.utils to help compute them  as below


```
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Convert to dictionary format
class_weight = dict(zip(np.unique(y_train), class_weights))
print(class_weight)
```
Next, we add these class weights during the training of our model. To compare the performance, we add this class weights back to the logistic regression model as below and observe its performance thereafter. 
```
#Logistic regression with weights

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Train logistic regression model with class weights
model = LogisticRegression(class_weight=class_weight)
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
print(classification_report(y_test, y_pred))
# Metrics
print(accuracy_score(y_test, y_pred))
```
When these blocks of code are executed, the first thing we notice about this approach to handling class imbalance is an improvement in the model performance from 61% to 67% overall accuracy. What about improvement in predicting fraudulent claims correctly? Well, that has improved significantly as well as observed from the confusion matrix below;

![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/confusion3.png?raw=true)
In predicting fraudulent claims correctly, the new model has now predicted 105 claims correctly as fraudulent, which is an improvement over the previous 65 claims. This directly reduces the number of fraud claims misclassified as legit from 71 claims to 31 claims which is what we want. In general, there is an overall improvement in the performance of the class classification as observed from the previous confusion matrices. 

However, 31 misclassified claims could still be high when we consider the monetary values (say a fraud classified as legit could result in thousands of dollars).  


Next on the agenda is to experiment with XGBoost (Extreme Gradient Boosting). XGBoost's ability to handle imbalanced datasets, its high accuracy and robustness, speed, interpretability, flexibility, and scalability make it an excellent choice for fraud detection tasks. In handling the class imbalance for this setup, I use the imbalance module. This model allows us to either oversample the minority class(fraud cases) or underrsample the majority class(legit claims). I experiment with both and found the undersampling technique to work better. Note that the previous strategy of assigning wait classes could still suffice for XGBoost too.  The complete setup is below;

```
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Assuming you have already loaded and preprocessed your data into X and y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Perform undersampling on the training data
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [5, 10, 15, 25],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'n_estimators': [10, 20, 40, 60]
} 

# Initialize XGBoost classifier
xgb = XGBClassifier()

# Custom scorer for GridSearchCV
custom_scorer = make_scorer(f1_score, pos_label=1)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=custom_scorer, cv=5)

# Perform GridSearchCV
grid_search.fit(X_train_scaled, y_train_resampled)

# Get the best parameters found
best_params = grid_search.best_params_

# Print the best parameters found by GridSearchCV
print("Best Parameters:", best_params)


# Initialize XGBoost classifier with the best parameters and class weights
xgb_best = XGBClassifier(**best_params)

# Train the classifier on the entire training data
xgb_best.fit(X_train_scaled, y_train_resampled)

# Predict on the test set
y_pred = xgb_best.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))

```
![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/report.png?raw=true)


```
# Calculate confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
class_names = ['Legit', 'Fraud']
# Plot confusion matrix
plt.figure(figsize=(8, 6))
# Calculate class percentages
class_percentages = cm / cm.sum(axis=1)[:, np.newaxis]
# Plot confusion matrix with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Class Counts and Percentages)')
plt.xticks(ticks=np.arange(2) + 0.5, labels=class_names)
plt.yticks(ticks=np.arange(2) + 0.5, labels=class_names)

# Add text annotations for class percentages
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # Compute percentage if count is not zero
        if cm[i, j] != 0:
            percentage = class_percentages[i, j]
            plt.text(j + 0.5, i + 0.2, f'{percentage:.2%}', 
                     horizontalalignment='center', verticalalignment='center', color='green')

plt.show()
```

![data info](https://github.com/BrightABOH/BrightABOH.github.io/blob/gh-pages/photos/confusionF.png?raw=true)
The results from the confusion matrix indicates predictions are doing better as compared to the previous results from the other algorithms. We are currently misclassifying 17 fraud claims as legit (even though still high)  is a huge reduction from the 31 previously. In addition, the we predicted 119 fraud claims correctly, which is an increase from the previous 105. Overall, we see an improvement in the model performance across different metrics measured.
