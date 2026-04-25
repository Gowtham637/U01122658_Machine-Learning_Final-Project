import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepareData():

    diabetesData = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

    print(diabetesData.head())
    print(diabetesData.info())
    print(diabetesData["Diabetes_012"].value_counts())
    print(diabetesData.isna().sum())

    duplicateRows = diabetesData.duplicated().sum()
    print("Number of duplicate rows:", duplicateRows)

    diabetesData = diabetesData.drop_duplicates()

    print(diabetesData.duplicated().sum())
    print(diabetesData["Diabetes_012"].value_counts())

    normalAndDiabetesData = diabetesData[diabetesData["Diabetes_012"] != 1].copy()

    inputData = normalAndDiabetesData.drop("Diabetes_012", axis=1)

    outputData = normalAndDiabetesData["Diabetes_012"].replace({
        0: 0,
        2: 1
    })

    trainingInputData, testingInputData, trainingOutputData, testingOutputData = train_test_split(
        inputData,
        outputData,
        test_size=0.2,
        random_state=42
    )

    smoteTechnique = SMOTE(random_state=42)

    balancedTrainingInputData, balancedTrainingOutputData = smoteTechnique.fit_resample(
        trainingInputData,
        trainingOutputData
    )

    print("Before balancing:")
    print(trainingOutputData.value_counts())

    print("After balancing:")
    print(balancedTrainingOutputData.value_counts())

    return balancedTrainingInputData, testingInputData, balancedTrainingOutputData, testingOutputData