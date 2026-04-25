from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def trainLogisticRegressionModel(balancedTrainingInputData, balancedTrainingOutputData):

    diabetesDirectionModel = LogisticRegression(max_iter=3000)

    diabetesDirectionModel.fit(
        balancedTrainingInputData,
        balancedTrainingOutputData
    )

    return diabetesDirectionModel

def trainRandomForestClassifier(balancedTrainingInputData,balancedTrainingOutputData):

  diabetesDirectionModel = RandomForestClassifier(
      n_estimators=200,
      max_depth=8,
      random_state=42
    )

  diabetesDirectionModel.fit(
    balancedTrainingInputData,
    balancedTrainingOutputData
   )

  return diabetesDirectionModel



def trainXGBClassifier(balancedTrainingInputData, balancedTrainingOutputData):

 diabetesDirectionModel = XGBClassifier(
      n_estimators=200,
      max_depth=5,
      learning_rate=0.1,
      random_state=42,
      eval_metric="logloss"
  )

 diabetesDirectionModel.fit(
    balancedTrainingInputData,
    balancedTrainingOutputData
)
 
 return diabetesDirectionModel