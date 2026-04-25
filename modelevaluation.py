from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def evaluateLogisticRegressionModel(diabetesDirectionModel, testingInputData, testingOutputData):

    testingPredictionData = diabetesDirectionModel.predict(testingInputData)

    print(classification_report(testingOutputData, testingPredictionData))
    print(confusion_matrix(testingOutputData, testingPredictionData))

    return testingPredictionData