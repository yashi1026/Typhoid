import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("typhoid_model.pkl", "rb"))

st.title("🦠 Typhoid Disease Prediction System")

st.write("Enter the patient details below:")

# Example inputs (Change according to your model features)
age = st.number_input("Age", min_value=0, max_value=120, value=25)
fever = st.selectbox("High Fever (0 = No, 1 = Yes)", [0, 1])
weakness = st.selectbox("Weakness (0 = No, 1 = Yes)", [0, 1])
vomiting = st.selectbox("Vomiting (0 = No, 1 = Yes)", [0, 1])
abdominal_pain = st.selectbox("Abdominal Pain (0 = No, 1 = Yes)", [0, 1])

# Prediction button
if st.button("Predict"):
    input_data = np.array([[age, fever, weakness, vomiting, abdominal_pain]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Patient is likely to have Typhoid.")
    else:
        st.success("✅ Patient is not likely to have Typhoid.")