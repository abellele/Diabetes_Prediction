import streamlit as st
import numpy as np
import joblib

# Load model
with open("model.pkl", "rb") as file:
    model = joblib.load("model.pkl")

# Judul aplikasi
st.title("Sugar Sense")

st.write("Put Your Data Here")

# Input user
glucose = st.number_input("Glucose Level", min_value=0.0)
hypertensive = st.selectbox("Hypertensive", [0, 1])
pulse_rate = st.number_input("Pulse Rate", min_value=0.0)
weight = st.number_input("Weight (kg)", min_value=0.0)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0)

# Tombol prediksi
if st.button("Predict"):

    # Susun input sesuai urutan training model
    input_data = np.array([[
        glucose,
        hypertensive,
        pulse_rate,
        weight,
        systolic_bp,
        bmi,
        diastolic_bp
    ]])

    # Prediksi
    prediction = model.predict(input_data)[0]

    # Jika model punya predict_proba
    try:
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of Diabetes: {probability:.2f}")
    except:
        probability = None

    # Hasil
    if prediction == 1:
        st.error("Patient is predicted to have Diabetes")
    else:
        st.success("Patient is predicted not to have Diabetes")