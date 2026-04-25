import pandas as pd

def predictPrediabetesRisk(diabetesDirectionModel):

    diabetesData = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

    prediabetesData = diabetesData[diabetesData["Diabetes_012"] == 1].copy()

    prediabetesInputData = prediabetesData.drop("Diabetes_012", axis=1)

    probabilityData = diabetesDirectionModel.predict_proba(prediabetesInputData)

    prediabetesData["normalSideProbability"] = probabilityData[:, 0]
    prediabetesData["diabetesSideProbability"] = probabilityData[:, 1]

    prediabetesData["riskDirection"] = "low risk toward normal"

    prediabetesData.loc[
        prediabetesData["diabetesSideProbability"] >= 0.5,
        "riskDirection"
    ] = "high risk toward diabetes"

    print(prediabetesData[[
        "Diabetes_012",
        "normalSideProbability",
        "diabetesSideProbability",
        "riskDirection"
    ]].head())

    print(prediabetesData["riskDirection"].value_counts())

    return prediabetesData