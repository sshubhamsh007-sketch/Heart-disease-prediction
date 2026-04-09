import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load model files
model = joblib.load("Logistic_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Title
st.title("Heart Disease Prediction by Shubham ❤️")
st.markdown("Provide the following details")

# Inputs
age = st.slider("Age", 18, 100, 40)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

max_hr = st.slider(
    "Max Heart Rate",
    60,
    220,
    150
)

old_peak = st.slider(
    "Old Peak (ST depression)",
    0.0,
    6.0,
    1.0
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# Prediction Button
if st.button("Predict"):

    # Raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": old_peak,   

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    
    input_df = pd.DataFrame([raw_input])

    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    
    input_df = input_df[expected_columns]

    
    numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    
    prediction = model.predict(input_df)[0]

    # Output
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")