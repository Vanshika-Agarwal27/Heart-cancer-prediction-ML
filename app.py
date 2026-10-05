import os
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "heart_disease_model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction using Machine Learning")
st.write("An educational ML application that estimates the class predicted by a trained Random Forest model.")

st.sidebar.header("Patient Information")
age=st.sidebar.slider("Age",20,90,50)
sex=st.sidebar.selectbox("Sex",["Female","Male"])
chest=st.sidebar.selectbox("Chest Pain Type",[0,1,2,3])
bp=st.sidebar.slider("Resting Blood Pressure",80,220,130)
chol=st.sidebar.slider("Cholesterol",100,450,220)
sugar=st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
hr=st.sidebar.slider("Maximum Heart Rate",60,220,150)
angina=st.sidebar.selectbox("Exercise-Induced Angina",[0,1])
oldpeak=st.sidebar.slider("ST Depression (Oldpeak)",0.0,6.0,1.0,0.1)
slope=st.sidebar.selectbox("ST Slope",[0,1,2])

input_df=pd.DataFrame([{
    "age":age,
    "sex":1 if sex=="Male" else 0,
    "chest_pain_type":chest,
    "resting_blood_pressure":bp,
    "cholesterol":chol,
    "fasting_blood_sugar":sugar,
    "maximum_heart_rate":hr,
    "exercise_induced_angina":angina,
    "oldpeak":oldpeak,
    "st_slope":slope
}])

if st.button("Predict", type="primary"):
    result=int(model.predict(input_df)[0])
    probability=float(model.predict_proba(input_df)[0][1])
    if result==1:
        st.warning(f"Model prediction: higher-risk class ({probability*100:.1f}% estimated probability)")
    else:
        st.success(f"Model prediction: lower-risk class ({(1-probability)*100:.1f}% estimated probability)")

st.subheader("Machine Learning Workflow")
st.write("Dataset → Train/Test Split → Random Forest Classifier → Prediction")

st.info("Educational demonstration only. This model is not a medical diagnostic tool. Real medical decisions should be made by qualified healthcare professionals.")
