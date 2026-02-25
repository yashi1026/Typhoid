import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Settings
st.set_page_config(page_title="Typhoid Prediction", layout="centered")

# Safe Model Loading for Streamlit Cloud
@st.cache_resource
def load_model():
    with open("typhoid_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("🦠 Typhoid Disease Prediction System")
st.markdown("### Enter Patient Details")

# Sidebar Patient Info
st.sidebar.header("Patient Information")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Symptoms
st.subheader("Symptoms")

col1, col2 = st.columns(2)

with col1:
    fever = st.selectbox("High Fever", [0,1])
    weakness = st.selectbox("Weakness", [0,1])
    vomiting = st.selectbox("Vomiting", [0,1])

with col2:
    abdominal_pain = st.selectbox("Abdominal Pain", [0,1])
    headache = st.selectbox("Headache", [0,1])
    fatigue = st.selectbox("Fatigue", [0,1])

input_data = np.array([[age, fever, weakness,
                        vomiting, abdominal_pain,
                        headache, fatigue]])

# Predict Button
if st.button("Predict Typhoid"):

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = 0.5

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"{name} is Likely to Have Typhoid")
    else:
        st.success(f"{name} is Not Likely to Have Typhoid")

    st.subheader("Disease Probability")
    st.progress(float(probability))

    # Risk Level
    if probability < 0.3:
        st.success("Low Risk")
    elif probability < 0.7:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

    # Download Report
    report = pd.DataFrame({
        "Name":[name],
        "Age":[age],
        "Gender":[gender],
        "Fever":[fever],
        "Weakness":[weakness],
        "Vomiting":[vomiting],
        "Abdominal Pain":[abdominal_pain],
        "Headache":[headache],
        "Fatigue":[fatigue],
        "Prediction":[prediction],
        "Probability":[probability]
    })

    st.download_button(
        label="Download Report",
        data=report.to_csv(index=False),
        file_name="Typhoid_Report.csv",
        mime="text/csv"
    )
