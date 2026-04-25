import streamlit as st
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from preprocessing import prepareData
from modeltraining import trainLogisticRegressionModel
from modeltraining import trainRandomForestClassifier
from modeltraining import trainXGBClassifier


st.set_page_config(
    page_title=" Diabetes Risk Prediction Project ",
    layout="wide"
)

st.title("PREDECTING A PERSON WILL BE NORMAL OR DIABETIES IN FUTURE")

st.write(
    "This project compares machine learning models using no-diabetes and diabetes cases. "
    "Then it checks whether a prediabetes person looks closer to the normal group or diabetes group."
)


diabetesData = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")


st.header("Dataset Overview")

st.write("Dataset shape:")
st.write(diabetesData.shape)

st.write("First five rows:")
st.dataframe(diabetesData.head())

st.write("Target distribution:")
st.write(diabetesData["Diabetes_012"].value_counts())

st.header(" TRAING MODELS AND COMPARING THEM ")

if st.button("Run Models"):

    balancedTrainingInputData, testingInputData, balancedTrainingOutputData, testingOutputData = prepareData()

    modelList = [
        ("Logistic Regression", trainLogisticRegressionModel),
        ("Random Forest", trainRandomForestClassifier),
        ("XGBoost", trainXGBClassifier)
    ]

    finalResults = []

    for modelName, modelFunction in modelList:

        st.subheader(modelName)

        model = modelFunction(
            balancedTrainingInputData,
            balancedTrainingOutputData
        )

        prediction = model.predict(testingInputData)

        accuracy = accuracy_score(testingOutputData, prediction)
        precision = precision_score(testingOutputData, prediction)
        recall = recall_score(testingOutputData, prediction)
        f1 = f1_score(testingOutputData, prediction)

        st.write("Confusion Matrix")
        st.write(confusion_matrix(testingOutputData, prediction))

        finalResults.append({
            "Model": modelName,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    resultTable = pd.DataFrame(finalResults)

    st.subheader("Model Comparison")
    st.dataframe(resultTable)

    bestModel = resultTable.sort_values(by="Recall", ascending=False).iloc[0]

    st.success("Best model based on recall: " + bestModel["Model"])


st.header("Prediabetes Risk Direction")

st.write("Enter values for one person.")

highBP = st.selectbox("High BP", [0, 1])
highChol = st.selectbox("High Cholesterol", [0, 1])
cholCheck = st.selectbox("Cholesterol Check", [0, 1])

bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0)

smoker = st.selectbox("Smoker", [0, 1])
stroke = st.selectbox("Stroke", [0, 1])
heartDisease = st.selectbox("Heart Disease or Attack", [0, 1])
physicalActivity = st.selectbox("Physical Activity", [0, 1])
fruits = st.selectbox("Fruits", [0, 1])
veggies = st.selectbox("Veggies", [0, 1])
heavyAlcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1])
anyHealthcare = st.selectbox("Any Healthcare", [0, 1])
noDoctorCost = st.selectbox("No Doctor Because of Cost", [0, 1])

generalHealth = st.slider("General Health", 1, 5, 3)
mentalHealth = st.slider("Mental Health Days", 0, 30, 0)
physicalHealth = st.slider("Physical Health Days", 0, 30, 0)

difficultyWalking = st.selectbox("Difficulty Walking", [0, 1])
sex = st.selectbox("Sex", [0, 1])
age = st.slider("Age Category", 1, 13, 8)
education = st.slider("Education", 1, 6, 4)
income = st.slider("Income", 1, 8, 5)


if st.button("Check Risk Direction"):

    balancedTrainingInputData, testingInputData, balancedTrainingOutputData, testingOutputData = prepareData()

    diabetesDirectionModel = trainLogisticRegressionModel(
        balancedTrainingInputData,
        balancedTrainingOutputData
    )

    personData = pd.DataFrame([{
        "HighBP": highBP,
        "HighChol": highChol,
        "CholCheck": cholCheck,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heartDisease,
        "PhysActivity": physicalActivity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": heavyAlcohol,
        "AnyHealthcare": anyHealthcare,
        "NoDocbcCost": noDoctorCost,
        "GenHlth": generalHealth,
        "MentHlth": mentalHealth,
        "PhysHlth": physicalHealth,
        "DiffWalk": difficultyWalking,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income
    }])

    probability = diabetesDirectionModel.predict_proba(personData)

    normalChance = probability[0][0]
    diabetesChance = probability[0][1]

    st.write("Normal side probability:", normalChance)
    st.write("Diabetes side probability:", diabetesChance)

    if diabetesChance >= 0.5:
        st.error("High risk toward diabetes")
    else:
        st.success("Low risk toward normal")


st.header("Project Limitation")

st.write(
    "This project does not predict the exact future because the dataset has no time-based follow-up records. "
    "It checks whether the current health pattern looks closer to no diabetes or diabetes."
)