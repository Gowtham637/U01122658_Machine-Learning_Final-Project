

# U01122658_Machine-Learning_Final-Project
# Diabetes Risk Prediction using CDC Health Indicators
## Overview

For this project, I worked with a real-world healthcare dataset from the CDC (BRFSS).
The goal was to understand how different health and lifestyle factors relate to diabetes and build a model that can analyze risk.

Instead of directly predicting all three classes, I took a slightly different approach. I trained models on clear cases (no diabetes and diabetes) and then applied them to prediabetes cases to see which side they are closer to.


## Problem
The dataset has three categories:
* 0 → No diabetes
* 1 → Prediabetes
* 2 → Diabetes

Prediabetes is not a clear category, so training a model directly on all three classes can be confusing.

So instead of treating it as a normal multiclass problem, I:

* Trained models using only 0 and 2 (clear cases)
* Converted it into a binary classification problem
* Used the trained model to analyze prediabetes cases (class 1)

This way, the model acts more like a **pattern comparison system**.


## Dataset

Dataset source:
[https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The dataset is based on survey responses and includes information like:

* BMI
* Age
* Blood pressure
* Cholesterol
* Physical activity
* Lifestyle factors


Target column:

* Diabetes_012 (0, 1, 2)

## What I did

### Data preprocessing

* Loaded and checked the dataset
* Removed duplicate rows
* Filtered out prediabetes cases for training
* Converted the target into binary (0 and 2 → 0 and 1)
* Split data into training and testing

### Handling imbalance

The dataset was imbalanced (more normal cases than diabetes).

To fix this, I used SMOTE to balance the training data so the model doesn’t become biased.

### Models used

I tried a few different models:

* Logistic Regression
* Random Forest
* XGBoost

### Evaluation

I didn’t just rely on accuracy. I also looked at:

* Precision
* Recall
* F1-score
* Confusion matrix

I focused more on **recall**, because missing a high-risk case is more serious than a false prediction.

### Model comparison

All models were trained on the same data and compared using the same metrics.

The best model was selected based on recall.
### Prediabetes analysis

After training, I applied the model to prediabetes cases.

For each person, the model gives:

* Probability of being closer to normal
* Probability of being closer to diabetes

Based on that:

* If diabetes probability is higher → high risk
* Otherwise → low risk


## Application

I built a Streamlit app for this project.

The app allows:

* Viewing dataset overview
* Training and comparing models
* Entering user input
* Getting risk direction output


## Limitations

This dataset does not have time-based data.

So the model does not predict whether someone *will* get diabetes in the future.
It only checks whether their current health profile looks closer to diabetes or normal.


## Tools used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit


## Final thoughts

This project helped me understand how to work with real-world health data, handle imbalance, and think beyond standard classification problems by changing the problem approach.