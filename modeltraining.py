from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
 # traing logistic Regression model 
def trainLogisticRegressionModel(balancedTrainingInputData, balancedTrainingOutputData):

    diabetesDirectionModel = LogisticRegression(max_iter=3000)
     #fit model on balanced training data
    diabetesDirectionModel.fit(
        balancedTrainingInputData,
        balancedTrainingOutputData
    )

    return diabetesDirectionModel
# training Random Forest model
def trainRandomForestClassifier(balancedTrainingInputData,balancedTrainingOutputData):

  diabetesDirectionModel = RandomForestClassifier(
      n_estimators=200,
      max_depth=8,
      random_state=42
    )
   # fit model on balanced training data
  diabetesDirectionModel.fit(
    balancedTrainingInputData,
    balancedTrainingOutputData
   )

  return diabetesDirectionModel


#traing 3rd modelXgboostclassifier  model
def trainXGBClassifier(balancedTrainingInputData, balancedTrainingOutputData):

 diabetesDirectionModel = XGBClassifier(
      n_estimators=200,
      max_depth=5,
      learning_rate=0.1,
      random_state=42,
      eval_metric="logloss"
  )
   # fit model on balanced training data
 diabetesDirectionModel.fit(
    balancedTrainingInputData,
    balancedTrainingOutputData
)
 
 return diabetesDirectionModel