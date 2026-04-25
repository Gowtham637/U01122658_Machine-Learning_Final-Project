from preprocessing import prepareData
from modeltraining import trainLogisticRegressionModel, trainRandomForestClassifier, trainXGBClassifier
from modelevaluation import evaluateLogisticRegressionModel
from prediction import predictPrediabetesRisk


# Step 1: load and prepare data
balancedTrainingInputData, testingInputData, balancedTrainingOutputData, testingOutputData = prepareData()


# Step 2: Logistic Regression
print("\n========== Logistic Regression ==========")

diabetesDirectionModel = trainLogisticRegressionModel(
    balancedTrainingInputData,
    balancedTrainingOutputData
)

testingPredictionData = evaluateLogisticRegressionModel(
    diabetesDirectionModel,
    testingInputData,
    testingOutputData
)


# Step 3: Random Forest
print("\n========== Random Forest ==========")

randomForestModel = trainRandomForestClassifier(
    balancedTrainingInputData,
    balancedTrainingOutputData
)

evaluateLogisticRegressionModel(
    randomForestModel,
    testingInputData,
    testingOutputData
)


# Step 4: XGBoost
print("\n========== XGBoost ==========")

xgboostModel = trainXGBClassifier(
    balancedTrainingInputData,
    balancedTrainingOutputData
)

evaluateLogisticRegressionModel(
    xgboostModel,
    testingInputData,
    testingOutputData
)


# Step 5: Prediabetes prediction (using Logistic model)
print("\n========== Prediabetes Risk Prediction ==========")

prediabetesResult = predictPrediabetesRisk(diabetesDirectionModel)