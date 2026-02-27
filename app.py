# ==============================
# Typhoid Prediction Streamlit App
# ==============================

import streamlit as st
import numpy as np
import joblib

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Typhoid Disease Prediction", layout="centered")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    model = joblib.load("typhoid_model.pkl")
    return model

model = load_model()

# -------- TITLE --------
st.title("🦠 Typhoid Disease Prediction System")
st.write("Enter the symptoms below to check whether the patient may have Typhoid.")

# -------- INPUT FIELDS --------
fever = st.number_input("Fever (0 = No, 1 = Yes)", min_value=0, max_value=1)
headache = st.number_input("Headache (0 = No, 1 = Yes)", min_value=0, max_value=1)
abdominal_pain = st.number_input("Abdominal Pain (0 = No, 1 = Yes)", min_value=0, max_value=1)
diarrhea = st.number_input("Diarrhea (0 = No, 1 = Yes)", min_value=0, max_value=1)
vomiting = st.number_input("Vomiting (0 = No, 1 = Yes)", min_value=0, max_value=1)
fatigue = st.number_input("Fatigue (0 = No, 1 = Yes)", min_value=0, max_value=1)
loss_of_appetite = st.number_input("Loss of Appetite (0 = No, 1 = Yes)", min_value=0, max_value=1)

# -------- PREDICTION --------
if st.button("Predict"):
    
    input_data = np.array([[fever, headache, abdominal_pain,
                            diarrhea, vomiting, fatigue,
                            loss_of_appetite]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Typhoid Disease")
    else:
        st.success("✅ The patient is unlikely to have Typhoid Disease")
