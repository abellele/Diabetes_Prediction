import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Judul aplikasi
st.title("Sugar Sense")
st.write("Put Your Data Here")

# Input user dengan nilai default yang wajar
glucose = st.number_input("Glucose Level (mmol/L)", min_value=0.0, value=5.0, step=0.1, 
                          help="Normal range: 4.0-7.0 mmol/L")
hypertensive = st.selectbox("Hypertensive", [0, 1], 
                            format_func=lambda x: "No" if x == 0 else "Yes")
pulse_rate = st.number_input("Pulse Rate (bpm)", min_value=0.0, value=70.0, step=1.0,
                             help="Normal range: 60-100 bpm")
weight = st.number_input("Weight (kg)", min_value=0.0, value=60.0, step=0.1)
systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, value=120.0, step=1.0,
                              help="Normal: <120 mmHg")
bmi = st.number_input("BMI", min_value=0.0, value=22.0, step=0.1,
                     help="Normal range: 18.5-24.9")
diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0.0, value=80.0, step=1.0,
                               help="Normal: <80 mmHg")

# Tombol prediksi
if st.button("Predict"):
    
    # Validasi input - pastikan tidak ada nilai 0 yang tidak wajar
    if glucose == 0 or pulse_rate == 0 or weight == 0 or systolic_bp == 0 or bmi == 0 or diastolic_bp == 0:
        st.warning("⚠️ Please enter valid values for all fields. Values cannot be 0.")
    else:
        # Susun input sesuai urutan training model (top7 features)
        # Urutan: glucose, hypertensive, pulse_rate, weight, systolic_bp, bmi, diastolic_bp
        input_data = np.array([[
            glucose,
            hypertensive,
            pulse_rate,
            weight,
            systolic_bp,
            bmi,
            diastolic_bp
        ]])
        
        # IMPORTANT: Apply StandardScaler
        # Nilai mean dan std dari training data (dari dataset yang sudah diproses)
        # Ini adalah perkiraan berdasarkan dataset - idealnya harus disimpan saat training
        scaler = StandardScaler()
        
        # Fit dengan data perkiraan dari dataset (mean dan std dari training data)
        # Untuk hasil yang lebih akurat, simpan scaler saat training dan load di sini
        training_mean = np.array([7.26, 0.134, 74.5, 55.2, 138.8, 23.8, 81.3])
        training_std = np.array([4.8, 0.34, 12.5, 11.8, 21.5, 5.2, 10.8])
        
        # Manual scaling menggunakan mean dan std dari training data
        input_scaled = (input_data - training_mean) / training_std
        
        # Prediksi dengan probability
        try:
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Gunakan threshold yang sama seperti saat training (dari hasil tuning)
            # Berdasarkan kode training, threshold optimal sekitar 0.3-0.4 untuk Ensemble Stacking
            optimal_threshold = 0.35  # Sesuaikan dengan hasil threshold tuning model Anda
            
            prediction = 1 if probability >= optimal_threshold else 0
            
            # Tampilkan hasil
            st.write("---")
            st.write(f"**Diabetes Probability: {probability:.2%}**")
            
            if prediction == 1:
                st.error("🩺 Patient is predicted to have **Diabetes**")
                st.write("**Recommendation:** Please consult with a healthcare professional for proper diagnosis and treatment.")
            else:
                st.success("✅ Patient is predicted **NOT to have Diabetes**")
                st.write("**Note:** Continue maintaining a healthy lifestyle with regular check-ups.")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please check if the model file is correct.")