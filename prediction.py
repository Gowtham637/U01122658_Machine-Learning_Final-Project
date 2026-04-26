import pandas as pd
# this function uses trained model to analyze prediabetes cases
def predictPrediabetesRisk(diabetesDirectionModel):
    # load dataset for predicting prediabetes records
    diabetesData = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

    #selecting the prediabeties (class1)
    prediabetesData = diabetesData[diabetesData["Diabetes_012"] == 1].copy()
    # remove target column to create input features
    prediabetesInputData = prediabetesData.drop("Diabetes_012", axis=1)

    # get probability of each case being normal or diabetes
    probabilityData = diabetesDirectionModel.predict_proba(prediabetesInputData)

    # store probabilities in dataframe for analysis
    prediabetesData["normalSideProbability"] = probabilityData[:, 0]
    prediabetesData["diabetesSideProbability"] = probabilityData[:, 1]
   #assuming towards low disk direction
    prediabetesData["riskDirection"] = "low risk toward normal"
    # updating risk if diabetes probability is higher
    prediabetesData.loc[
        prediabetesData["diabetesSideProbability"] >= 0.5,
        "riskDirection"
    ] = "high risk toward diabetes"
    # show sample output
    print(prediabetesData[[
        "Diabetes_012",
        "normalSideProbability",
        "diabetesSideProbability",
        "riskDirection"
    ]].head())
    # show overall risk distribution
    print(prediabetesData["riskDirection"].value_counts())

    return prediabetesData