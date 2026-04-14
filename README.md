

# U01122658_Machine-Learning_Final-Project
# Diabetes Risk Prediction using CDC Health Indicators
## Overview

For this project, I chose to work with a real-world healthcare dataset from the CDC (BRFSS). The main idea is to understand how different health and lifestyle factors are related to diabetes and see if I can build a model that predicts diabetes risk.
Right now, the project is in the initial stage, so the focus is on exploring the data properly before moving into modeling.

## Problem
Diabetes is a very common health condition, and in many cases it can be managed better if identified early.
In this project, the goal is to use the given health data to classify people into:
* No diabetes
* Prediabetes
* Diabetes
So this becomes a multiclass classification problem.

## Dataset

Dataset source:
[https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The dataset is based on survey responses and includes information like:

* BMI
* Age
* Blood pressure
* Physical activity
* Smoking habits
* General health

Target column:

* Diabetes_012 (0, 1, 2)

## What I plan to do

Since I am just starting, I will be following a step-by-step approach instead of jumping directly into modeling.

### Step 1: Understand the data

First, I will look at the dataset structure, number of rows and columns, and data types. This will help me get a basic idea of what I am working with.

### Step 2: Explore the data (EDA)

I plan to check how the data is distributed and look for patterns.

Some things I will focus on:

* How the diabetes categories are distributed
* How features like BMI and age vary
* Any visible relationships between features

### Step 3: Clean the data

If there are missing values or inconsistencies, I will handle them before moving forward.

### Step 4: Preprocess the data

This includes:

* Converting categorical values into numbers
* Scaling numerical features if needed
* Preparing the dataset for model training

### Step 5: Handle imbalance

If the dataset is imbalanced, I will try basic techniques to handle it so that the model does not become biased.

### Step 6: Build models

I plan to try a few different models such as:

* Logistic Regression
* Random Forest
* Gradient Boosting

### Step 7: Evaluate models

Instead of only looking at accuracy, I will also check:

* Precision
* Recall
* F1-score
* Confusion matrix

### Step 8: Compare results

After training different models, I will compare their performance and see which one works better.

## Expected outcome

By the end of this project, I expect to:

* Understand which factors are most related to diabetes
* Build a basic model that can predict diabetes risk
* Get hands-on experience with a real dataset

## Tools

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Jupyter Notebook
## Note

This project is still in progress, and I will be updating this file as I complete each step.

