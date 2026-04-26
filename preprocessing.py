import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepareData():
     # load data set for preprocessing 
    diabetesData = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
      # basic checks 
    print(diabetesData.head())
    print(diabetesData.info())
    print(diabetesData["Diabetes_012"].value_counts())
    print(diabetesData.isna().sum())
      # identify and remove duplicate rows to avoid bias
    duplicateRows = diabetesData.duplicated().sum()
    print("Number of duplicate rows:", duplicateRows)

    diabetesData = diabetesData.drop_duplicates()

    print(diabetesData.duplicated().sum())
    print(diabetesData["Diabetes_012"].value_counts())
    # remove prediabetes (class 1) to train only on clear cases
    normalAndDiabetesData = diabetesData[diabetesData["Diabetes_012"] != 1].copy()
     #seprating input and target variable 
    inputData = normalAndDiabetesData.drop("Diabetes_012", axis=1)
      # converting  target into binary (0 = normal, 1 = diabetes) as im focusing on pre diabeties 
    outputData = normalAndDiabetesData["Diabetes_012"].replace({
        0: 0,
        2: 1
    })
       # split data into traning and testing dataset 
    trainingInputData, testingInputData, trainingOutputData, testingOutputData = train_test_split(
        inputData,
        outputData,
        test_size=0.2,
        random_state=42
    )
       #this dataset is imbalance so applying the smote technique to handle   class imbalance 
    smoteTechnique = SMOTE(random_state=42)

    balancedTrainingInputData, balancedTrainingOutputData = smoteTechnique.fit_resample(
        trainingInputData,
        trainingOutputData
    )

    print("Before balancing:")
    print(trainingOutputData.value_counts())

    print("After balancing:")
    print(balancedTrainingOutputData.value_counts())
       # return processed and balanced data

    return balancedTrainingInputData, testingInputData, balancedTrainingOutputData, testingOutputData