import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
from translations import t

# Define feature order corresponding to the trained model
FEATURE_NAMES = ['glucose', 'hypertensive', 'systolic_bp', 'diastolic_bp', 'bmi', 'weight', 'pulse_rate']
OPTIMAL_THRESHOLD = 0.73  # Validation-tuned threshold for Ensemble Stacking (New Revision)

# Default fallback scaler parameters (from training set)
FALLBACK_MEAN = np.array([7.2629, 0.1105, 134.1616, 82.2852, 22.2454, 53.5635, 76.6386])
FALLBACK_SCALE = np.array([1.8129, 0.3135, 21.5387, 12.2587, 3.7197, 9.7586, 11.8925])

# Language selection
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

col_title, col_lang = st.columns([3, 1])
with col_lang:
    selected_lang_str = st.selectbox(
        t('lang_label', st.session_state.lang),
        ["English 🇬🇧", "Bahasa Indonesia 🇮🇩"],
        index=0 if st.session_state.lang == "en" else 1,
        key="simple_lang_selector"
    )
    chosen_lang = "en" if "English" in selected_lang_str else "id"
    if chosen_lang != st.session_state.lang:
        st.session_state.lang = chosen_lang
        st.rerun()

lang = st.session_state.lang

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None
    threshold = OPTIMAL_THRESHOLD

    if os.path.exists("model_metadata.json"):
        try:
            with open("model_metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                threshold = meta.get("optimal_threshold", OPTIMAL_THRESHOLD)
        except Exception:
            pass

    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
        except Exception as e:
            st.error(f"Error loading model.pkl: {e}")

    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception:
            scaler = None

    return model, scaler, threshold

model, scaler, optimal_threshold = load_model_and_scaler()

# Judul aplikasi
with col_title:
    st.title(t('app_title', lang))
    st.write(t('form_subtitle', lang))

# Input user dengan nilai default yang wajar
glucose = st.number_input(t('label_glucose', lang), min_value=0.0, value=5.0, step=0.1, 
                          help=t('help_glucose', lang))
hypertensive = st.selectbox(t('label_hypertensive', lang), [0, 1], 
                            format_func=lambda x: t('opt_no', lang) if x == 0 else t('opt_yes', lang))
systolic_bp = st.number_input(t('label_systolic', lang), min_value=0.0, value=120.0, step=1.0,
                              help=t('help_systolic', lang))
diastolic_bp = st.number_input(t('label_diastolic', lang), min_value=0.0, value=80.0, step=1.0,
                               help=t('help_diastolic', lang))
bmi = st.number_input(t('label_bmi', lang), min_value=0.0, value=22.0, step=0.1,
                     help=t('help_bmi', lang))
weight = st.number_input(t('label_weight', lang), min_value=0.0, value=60.0, step=0.1)
pulse_rate = st.number_input(t('label_pulse', lang), min_value=0.0, value=70.0, step=1.0,
                             help=t('help_pulse', lang))

# Tombol prediksi
if st.button(t('btn_predict', lang)):
    
    # Validasi input - pastikan tidak ada nilai 0 yang tidak wajar
    if glucose == 0 or pulse_rate == 0 or weight == 0 or systolic_bp == 0 or bmi == 0 or diastolic_bp == 0:
        st.warning(t('err_zero', lang))
    elif model is None:
        st.error(t('err_model', lang))
    else:
        # Susun DataFrame sesuai urutan training model baru:
        # ['glucose', 'hypertensive', 'systolic_bp', 'diastolic_bp', 'bmi', 'weight', 'pulse_rate']
        raw_df = pd.DataFrame([{
            'glucose': glucose,
            'hypertensive': hypertensive,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'bmi': bmi,
            'weight': weight,
            'pulse_rate': pulse_rate
        }])[FEATURE_NAMES]
        
        # Scaling
        if scaler is not None:
            input_scaled = scaler.transform(raw_df)
        else:
            input_scaled = (raw_df.values - FALLBACK_MEAN) / FALLBACK_SCALE
            
        input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURE_NAMES)
        
        # Prediksi dengan probability
        try:
            probability = float(model.predict_proba(input_scaled_df)[0][1])
            prediction = 1 if probability >= optimal_threshold else 0
            
            # Tampilkan hasil
            st.write("---")
            prob_label = "Diabetes Probability" if lang == 'en' else "Probabilitas Diabetes"
            thresh_label = "Decision Threshold" if lang == 'en' else "Ambang Batas Keputusan"
            st.write(f"**{prob_label}: {probability:.2%}** ({thresh_label}: {optimal_threshold:.0%})")
            
            if prediction == 1:
                st.error(f"🩺 {t('pos_title', lang)}")
                st.write(t('pos_highlight', lang))
            else:
                st.success(f"✅ {t('neg_title', lang)}")
                st.write(t('neg_highlight', lang))
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please check if the model file is correct.")