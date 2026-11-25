import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from translations import t

# Page configuration
st.set_page_config(
    page_title="SugarSense - Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global App Background */
    .stApp {
        background-color: #F4F7FE;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #2B3674;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    p, span, div {
        color: #718096;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero section Banner */
    .hero-section {
        position: relative;
        text-align: left;
        padding: 3rem 3rem;
        background: #2E66F6;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(46, 102, 246, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Banner background decoration */
    .hero-section::after {
        content: '';
        position: absolute;
        right: -5%;
        top: -50%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 2;
        color: #FFD700 !important; /* Gold */
    }
    
    .hero-subtitle {
        font-size: 18px;
        font-weight: 400;
        opacity: 0.9;
        position: relative;
        z-index: 2;
        color: #E2E8F0 !important;
    }

    /* Common Card Styling */
    .info-card, .chart-container, .input-container {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.04);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover, .chart-container:hover {
        transform: translateY(-4px);
        box-shadow: 0px 15px 35px rgba(0, 0, 0, 0.08);
    }
    
    .info-title {
        font-size: 24px;
        font-weight: 700;
        color: #2E66F6 !important; /* Royal Blue */
        margin-bottom: 1rem;
    }
    
    .info-text {
        font-size: 15px;
        color: #718096;
        line-height: 1.6;
    }
    
    /* Highlighted blue boxes inside cards */
    .highlight-box {
        background: #F4F7FE;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #FFF1F0;
        color: #E53E3E;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: left;
        margin: 2rem 0;
        font-size: 14px;
        border: 1px solid #FED7D7;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.04);
    }
    .disclaimer h3 {
        color: #DD6B20 !important; /* Orange */
        font-size: 18px;
        margin-bottom: 5px;
    }
    
    /* Input form styling */
    .input-title {
        text-align: left;
        font-size: 28px;
        font-weight: 700;
        color: #2E66F6 !important; /* Royal Blue */
        margin-bottom: 2rem;
    }

    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 0.5rem;
        background-color: #F8FAFC;
        transition: all 0.3s ease;
        color: #2B3674;
        font-weight: 500;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #2E66F6;
        box-shadow: 0 0 0 3px rgba(46, 102, 246, 0.15);
        background-color: #FFFFFF;
    }
    
    /* Result styling */
    .result-positive, .result-negative {
        background: #FFFFFF;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.04);
    }
    
    .result-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .result-subtitle {
        font-size: 16px;
        color: #718096;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: #2E66F6;
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 102, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(46, 102, 246, 0.4);
        background: #1C51D3;
        color: white !important;
    }

    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(46, 102, 246, 0.3);
    }
    
    .stButton>button p {
        color: white !important;
    }

    /* Sidebar specific styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Hide radio button circles in sidebar */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Style the radio text */
    [data-testid="stSidebar"] div[role="radiogroup"] p {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #A3AED0;
        transition: all 0.3s ease;
        padding: 10px 16px;
        border-radius: 12px;
        margin-bottom: 5px;
    }
    
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
        color: #2E66F6 !important;
        background-color: #F4F7FE;
    }
    
    /* Make active radio button blue */
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div + div p {
        color: #2E66F6 !important;
        background-color: #F4F7FE;
    }

    /* Media Queries */
    @media (max-width: 768px) {
        .hero-title { font-size: 32px; }
        .info-card, .input-container { padding: 1.5rem; }
        .result-title { font-size: 28px; }
    }
</style>
""", unsafe_allow_html=True)

# Model and Feature Configuration
FEATURE_NAMES = [
    'age', 'gender', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height',
    'weight', 'bmi', 'family_diabetes', 'hypertensive', 'family_hypertension',
    'cardiovascular_disease', 'stroke'
]
OPTIMAL_THRESHOLD = 0.72

FALLBACK_MEAN = np.array([
    45.8563, 0.7098, 76.6386, 134.1616, 82.2852, 7.2629, 1.5502,
    53.5635, 22.2454, 0.0327, 0.1105, 0.0351, 0.0105, 0.0041
])
FALLBACK_SCALE = np.array([
    13.3576, 0.4539, 11.8925, 21.5387, 12.2587, 1.8129, 0.0698,
    9.7586, 3.7197, 0.1778, 0.3135, 0.1841, 0.1021, 0.0635
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, and threshold
@st.cache_resource
def load_model_and_scaler():
    model = None
    scaler = None
    threshold = OPTIMAL_THRESHOLD
    metrics = {}

    meta_path = os.path.join(BASE_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                threshold = meta.get("optimal_threshold", OPTIMAL_THRESHOLD)
                metrics = meta.get("metrics", {})
        except Exception:
            pass

    model_path = os.path.join(BASE_DIR, "model.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = None

    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    return model, scaler, threshold, metrics

model, scaler, optimal_threshold, model_metrics = load_model_and_scaler()

def scale_input_features(raw_dict):
    df_in = pd.DataFrame([raw_dict])[FEATURE_NAMES]
    if scaler is not None:
        scaled_vals = scaler.transform(df_in)
    else:
        scaled_vals = (df_in.values - FALLBACK_MEAN) / FALLBACK_SCALE
    return pd.DataFrame(scaled_vals, columns=FEATURE_NAMES)

# Navigation - Sidebar
def render_sidebar():
    lang = st.session_state.get('lang', 'en')
    with st.sidebar:
        st.markdown("""
        <div style='text-align: left; padding: 1rem 0 1rem 1rem;'>
            <h1 style='color: #2B3674; font-size: 24px; font-weight: 800; display: flex; align-items: center; gap: 10px;'>
                <span style="background: #2E66F6; color: white; border-radius: 8px; padding: 5px 10px; font-size: 20px;">S</span>
                SugarSense
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Language Selector
        lang_options = ["English 🇬🇧", "Bahasa Indonesia 🇮🇩"]
        current_lang_idx = 0 if lang == "en" else 1
        chosen_str = st.selectbox(
            t('lang_label', lang),
            lang_options,
            index=current_lang_idx,
            key="app_language_selector"
        )
        selected_code = "en" if "English" in chosen_str else "id"
        if selected_code != lang:
            st.session_state.lang = selected_code
            st.rerun()

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
        
        nav_options = [t('nav_dashboard', st.session_state.lang), t('nav_prediction', st.session_state.lang)]
        current_nav_idx = 0 if st.session_state.page == "home" else 1
        selection = st.radio(
            "Navigation",
            nav_options,
            index=current_nav_idx,
            label_visibility="collapsed"
        )
        
        if selection == nav_options[0] and st.session_state.page != "home":
            st.session_state.page = "home"
            st.rerun()
        elif selection == nav_options[1] and st.session_state.page not in ["input", "result_negative", "result_positive"]:
            st.session_state.page = "input"
            st.rerun()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'lang' not in st.session_state:
    st.session_state.lang = "en"

# Home/About Page
def home_page():
    render_sidebar()
    lang = st.session_state.lang
    
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title"><span style="color: #FFD700;">{t('hero_title', lang)}</span></h1>
        <p class="hero-subtitle">{t('hero_subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h2 class="info-title"><span style="color: #2E66F6;">{t('what_is_title', lang)}</span></h2>
            <p class="info-text">
                {t('what_is_desc', lang)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h2 class="info-title"><span style="color: #2E66F6;">{t('what_do_title', lang)}</span></h2>
            <p class="info-text">
                {t('what_do_desc', lang)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h2 class="info-title"><span style="color: #2E66F6;">{t('why_title', lang)}</span></h2>
            <p class="info-text">
                {t('why_desc', lang)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="disclaimer">
            <h3><span style="color: #DD6B20;">{t('disclaimer_title', lang)}</span></h3>
            <p style="color: #718096; margin-top: 10px;">
                {t('disclaimer_desc', lang)}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(t('btn_start', lang), key="start_prediction", width="stretch"):
            st.session_state.page = "input"
            st.rerun()

# Input Page
def input_page():
    render_sidebar()
    lang = st.session_state.lang
    
    st.markdown(f"""
    <div class="input-container" style="max-width: 800px; margin: 2rem auto;">
        <h1 class="input-title"><span style="color: #2E66F6;">{t('form_title', lang)}</span></h1>
        <p style="color: #718096; margin-bottom: 2rem;">{t('form_subtitle', lang)}</p>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form", border=False):
        col1, col2 = st.columns([1, 1])
        yes_no = lambda x: t('opt_no', lang) if x == 0 else t('opt_yes', lang)
        
        with col1:
            age = st.number_input(t('label_age', lang), min_value=0, max_value=120, value=35, step=1, help=t('help_age', lang))
            gender = st.selectbox(t('label_gender', lang), options=[0, 1], format_func=lambda x: t('opt_male', lang) if x == 0 else t('opt_female', lang))
            height_cm = st.number_input(t('label_height', lang), min_value=0.0, value=160.0, step=1.0, help=t('help_height', lang))
            weight = st.number_input(t('label_weight', lang), min_value=0.0, value=60.0, step=0.1)
            bmi = st.number_input(t('label_bmi', lang), min_value=0.0, value=22.0, step=0.1, help=t('help_bmi', lang))
            glucose = st.number_input(t('label_glucose', lang), min_value=0.0, value=5.0, step=0.1, help=t('help_glucose', lang))
            pulse_rate = st.number_input(t('label_pulse', lang), min_value=0.0, value=70.0, step=1.0, help=t('help_pulse', lang))
            
        with col2:
            systolic_bp = st.number_input(t('label_systolic', lang), min_value=0.0, value=120.0, step=1.0, help=t('help_systolic', lang))
            diastolic_bp = st.number_input(t('label_diastolic', lang), min_value=0.0, value=80.0, step=1.0, help=t('help_diastolic', lang))
            hypertensive = st.selectbox(t('label_hypertensive', lang), options=[0, 1], format_func=yes_no)
            family_hypertension = st.selectbox(t('label_family_hypertension', lang), options=[0, 1], format_func=yes_no, help=t('help_family_hypertension', lang))
            family_diabetes = st.selectbox(t('label_family_diabetes', lang), options=[0, 1], format_func=yes_no, help=t('help_family_diabetes', lang))
            cardiovascular_disease = st.selectbox(t('label_cardiovascular', lang), options=[0, 1], format_func=yes_no, help=t('help_cardiovascular', lang))
            stroke = st.selectbox(t('label_stroke', lang), options=[0, 1], format_func=yes_no, help=t('help_stroke', lang))
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submit = st.form_submit_button(t('btn_predict', lang), width="stretch")
    
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if (age == 0 or height_cm == 0 or glucose == 0 or pulse_rate == 0 or weight == 0
                or systolic_bp == 0 or bmi == 0 or diastolic_bp == 0):
            st.error(t('err_zero', lang))
        elif model is None:
            st.error(t('err_model', lang))
        else:
            input_data = {
                'age': age,
                'gender': gender,
                'pulse_rate': pulse_rate,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'glucose': glucose,
                'height': height_cm / 100.0,
                'weight': weight,
                'bmi': bmi,
                'family_diabetes': family_diabetes,
                'hypertensive': hypertensive,
                'family_hypertension': family_hypertension,
                'cardiovascular_disease': cardiovascular_disease,
                'stroke': stroke
            }
            st.session_state.input_data = input_data
            scaled_df = scale_input_features(input_data)
            
            try:
                probability = float(model.predict_proba(scaled_df)[0][1])
                prediction = 1 if probability >= optimal_threshold else 0
                st.session_state.probability = probability
                st.session_state.prediction = prediction
                st.session_state.page = "result_positive" if prediction == 1 else "result_negative"
                st.rerun()
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

def render_interactive_insights(input_data):
    lang = st.session_state.lang
    st.markdown(f"<div class='info-card'><h3 style='background: linear-gradient(90deg, #3B82F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; font-weight: 800; display: inline-block;'>{t('insights_title', lang)}</h3>", unsafe_allow_html=True)
    
    tab_deep_dive, tab_simulator, tab_gauges = st.tabs([t('tab_deep_dive', lang), t('tab_simulator', lang), t('tab_gauges', lang)])
    
    with tab_deep_dive:
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        
        bmi = input_data.get('bmi', 0)
        if bmi < 18.5:
            bmi_color, bmi_bg, bmi_text, bmi_stat = "#F59E0B", "#FFFBEB", t('plan_underweight', lang), t('stat_underweight', lang)
        elif 25.0 <= bmi < 30.0:
            bmi_color, bmi_bg, bmi_text, bmi_stat = "#F59E0B", "#FFFBEB", t('plan_overweight', lang), t('stat_overweight', lang)
        elif bmi >= 30.0:
            bmi_color, bmi_bg, bmi_text, bmi_stat = "#EF4444", "#FEF2F2", t('plan_obese', lang), t('stat_obese', lang)
        else:
            bmi_color, bmi_bg, bmi_text, bmi_stat = "#10B981", "#ECFDF5", t('plan_healthy_weight', lang), t('stat_healthy_weight', lang)
            
        with col_a:
            st.markdown(f"""
            <div style="background-color: {bmi_bg}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {bmi_color}; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <h4 style="color: {bmi_color}; margin-top: 0; font-size: 18px;">{t('card_bmi_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 10px 0;">{bmi} <span style="font-size: 16px; font-weight: 600; color: {bmi_color};">({bmi_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {bmi_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        glucose = input_data.get('glucose', 0)
        if glucose > 7.0:
            gl_color, gl_bg, gl_text, gl_stat = "#F59E0B", "#FFFBEB", t('plan_above_normal', lang), t('stat_above_normal', lang)
        elif glucose < 4.0:
            gl_color, gl_bg, gl_text, gl_stat = "#F59E0B", "#FFFBEB", t('plan_below_normal', lang), t('stat_below_normal', lang)
        else:
            gl_color, gl_bg, gl_text, gl_stat = "#10B981", "#ECFDF5", t('plan_normal_glucose', lang), t('stat_normal_glucose', lang)
            
        with col_b:
            st.markdown(f"""
            <div style="background-color: {gl_bg}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {gl_color}; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <h4 style="color: {gl_color}; margin-top: 0; font-size: 18px;">{t('card_glucose_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 10px 0;">{glucose} <span style="font-size: 16px; font-weight: 600; color: {gl_color};">({gl_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {gl_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        col_c, col_d = st.columns(2)
        
        systolic = input_data.get('systolic_bp', 0)
        diastolic = input_data.get('diastolic_bp', 0)
        hypertensive = input_data.get('hypertensive', 0)
        if systolic >= 130 or diastolic >= 80 or hypertensive == 1:
            bp_color, bp_bg, bp_text, bp_stat = "#EF4444", "#FEF2F2", t('plan_elevated_bp', lang), t('stat_elevated_bp', lang)
        else:
            bp_color, bp_bg, bp_text, bp_stat = "#10B981", "#ECFDF5", t('plan_normal_bp', lang), t('stat_normal_bp', lang)
            
        with col_c:
            st.markdown(f"""
            <div style="background-color: {bp_bg}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {bp_color}; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <h4 style="color: {bp_color}; margin-top: 0; font-size: 18px;">{t('card_bp_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 10px 0;">{systolic}/{diastolic} <span style="font-size: 16px; font-weight: 600; color: {bp_color};">({bp_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {bp_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        pulse = input_data.get('pulse_rate', 0)
        if pulse > 100:
            hr_color, hr_bg, hr_text, hr_stat = "#F59E0B", "#FFFBEB", t('plan_high_hr', lang), t('stat_high_hr', lang)
        elif 0 < pulse < 60:
            hr_color, hr_bg, hr_text, hr_stat = "#8B5CF6", "#F5F3FF", t('plan_low_hr', lang), t('stat_low_hr', lang)
        else:
            hr_color, hr_bg, hr_text, hr_stat = "#10B981", "#ECFDF5", t('plan_normal_hr', lang), t('stat_normal_hr', lang)
            
        with col_d:
            st.markdown(f"""
            <div style="background-color: {hr_bg}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {hr_color}; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                <h4 style="color: {hr_color}; margin-top: 0; font-size: 18px;">{t('card_hr_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 10px 0;">{pulse} <span style="font-size: 16px; font-weight: 600; color: {hr_color};">({hr_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {hr_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
    with tab_gauges:
        st.markdown("<br>", unsafe_allow_html=True)
        def create_gauge(value, title, min_val, max_val, steps):
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title, 'font': {'size': 16, 'color': '#2B3674'}},
                gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': "#2B3674", 'thickness': 0.2}, 'steps': steps}
            ))
            val_text = f"{value:.1f}" if isinstance(value, (float, int)) and value % 1 != 0 else f"{int(value)}"
            fig.add_annotation(x=0.5, y=0.15, xref="paper", yref="paper", text=val_text, font=dict(size=32, color='#2B3674'), showarrow=False)
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
            return fig

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            bmi_steps = [{'range': [10, 18.5], 'color': '#FEEBC8'}, {'range': [18.5, 25], 'color': '#C6F6D5'},
                         {'range': [25, 30], 'color': '#FEEBC8'}, {'range': [30, 45], 'color': '#FED7D7'}]
            st.plotly_chart(create_gauge(input_data.get('bmi', 0), "BMI", 10, 45, bmi_steps), width="stretch")
            bp_steps = [{'range': [60, 120], 'color': '#C6F6D5'}, {'range': [120, 130], 'color': '#FEEBC8'}, {'range': [130, 200], 'color': '#FED7D7'}]
            st.plotly_chart(create_gauge(input_data.get('systolic_bp', 0), f"{t('label_systolic', lang)}", 60, 200, bp_steps), width="stretch")
            
        with col_g2:
            glucose_steps = [{'range': [0, 4.0], 'color': '#FEEBC8'}, {'range': [4.0, 7.0], 'color': '#C6F6D5'}, {'range': [7.0, 15.0], 'color': '#FED7D7'}]
            st.plotly_chart(create_gauge(input_data.get('glucose', 0), f"{t('label_glucose', lang)}", 0, 15, glucose_steps), width="stretch")
            pulse_steps = [{'range': [40, 60], 'color': '#FEEBC8'}, {'range': [60, 100], 'color': '#C6F6D5'}, {'range': [100, 160], 'color': '#FED7D7'}]
            st.plotly_chart(create_gauge(input_data.get('pulse_rate', 0), f"{t('label_pulse', lang)}", 40, 160, pulse_steps), width="stretch")

    with tab_simulator:
        st.markdown(f"<p style='color: #718096; margin-bottom: 1rem; margin-top: 1rem;'>{t('sim_desc', lang)}</p>", unsafe_allow_html=True)
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_bmi = st.slider("BMI", 10.0, 50.0, float(input_data.get('bmi', 22.0)), 0.5)
            sim_glucose = st.slider(t('label_glucose', lang), 2.0, 15.0, float(input_data.get('glucose', 5.0)), 0.1)
            sim_sys_bp = st.slider(t('label_systolic', lang), 80.0, 200.0, float(input_data.get('systolic_bp', 120.0)), 1.0)
        with sim_col2:
            sim_weight = st.slider(t('label_weight', lang), 30.0, 150.0, float(input_data.get('weight', 60.0)), 1.0)
            sim_dia_bp = st.slider(t('label_diastolic', lang), 50.0, 120.0, float(input_data.get('diastolic_bp', 80.0)), 1.0)
            sim_pulse = st.slider(t('label_pulse', lang), 40.0, 150.0, float(input_data.get('pulse_rate', 70.0)), 1.0)
            
        if model is not None:
            sim_input = dict(input_data)
            sim_input.update({
                'glucose': sim_glucose, 'systolic_bp': sim_sys_bp, 'diastolic_bp': sim_dia_bp,
                'bmi': sim_bmi, 'weight': sim_weight, 'pulse_rate': sim_pulse
            })
            sim_scaled_df = scale_input_features(sim_input)
            sim_prob = float(model.predict_proba(sim_scaled_df)[0][1]) * 100
            threshold_pct = optimal_threshold * 100
            is_above_thresh = sim_prob >= threshold_pct
            sim_msg = t('sim_above', lang, thresh=threshold_pct) if is_above_thresh else t('sim_below', lang, thresh=threshold_pct)
            
            st.markdown(f"""
            <div style="background: {'#FFF1F0' if is_above_thresh else '#F0FFF4'}; padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 1rem; border: 1px solid {'#FED7D7' if is_above_thresh else '#C6F6D5'};">
                <h4 style="color: {'#E53E3E' if is_above_thresh else '#38A169'}; margin-bottom: 0.5rem;">{t('sim_title', lang)}</h4>
                <h2 style="color: {'#E53E3E' if is_above_thresh else '#38A169'}; font-size: 32px; font-weight: 800; margin: 0;">{sim_prob:.1f}%</h2>
                <p style="color: #718096; margin-top: 0.5rem;">{sim_msg}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(t('sim_error', lang))

    st.markdown("</div>", unsafe_allow_html=True)


def render_result_explanation(probability, is_positive):
    """Show the risk score, decision threshold and model accuracy, and explain each in plain words."""
    lang = st.session_state.lang
    required = ('test_accuracy', 'test_precision', 'test_recall', 'test_samples', 'diabetes_share')
    if not all(k in model_metrics for k in required):
        return

    score_pct = probability * 100
    threshold_pct = optimal_threshold * 100
    accuracy_pct = model_metrics['test_accuracy'] * 100
    recall_pct = model_metrics['test_recall'] * 100
    precision_pct = model_metrics['test_precision'] * 100
    share_pct = model_metrics['diabetes_share'] * 100
    n_tested = int(model_metrics['test_samples'])

    def tile(color, bg, title, value, desc):
        return (
            f'<div style="background-color: {bg}; padding: 1.5rem; border-radius: 15px; '
            f'border-left: 5px solid {color}; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;">'
            f'<h4 style="color: {color}; margin-top: 0; font-size: 18px;">{title}</h4>'
            f'<p style="font-size: 32px; font-weight: 800; color: #1E293B; margin: 10px 0;">{value}</p>'
            f'<p style="color: #475569; font-size: 16px; margin-bottom: 0;">{desc}</p>'
            f'</div>'
        )

    st.markdown(
        f"<h3 style='background: linear-gradient(90deg, #3B82F6, #8B5CF6); -webkit-background-clip: text; "
        f"-webkit-text-fill-color: transparent; margin: 1.5rem 0 0.25rem 0; font-weight: 800; display: inline-block;'>"
        f"{t('explain_title', lang)}</h3>"
        f"<p style='color: #718096; margin-bottom: 1rem;'>{t('explain_intro', lang)}</p>",
        unsafe_allow_html=True
    )

    score_desc = t('explain_score_desc', lang)
    if not is_positive:
        score_desc += " " + t('explain_safe_note', lang)
    score_color, score_bg = ("#EF4444", "#FEF2F2") if is_positive else ("#10B981", "#ECFDF5")

    verdict_key = 'pos_title' if is_positive else 'neg_title'
    compare_key = 'explain_threshold_above' if is_positive else 'explain_threshold_below'
    threshold_desc = t('explain_threshold_desc', lang) + " " + t(compare_key, lang, score=score_pct, label=t(verdict_key, lang))
    accuracy_desc = t('explain_accuracy_desc', lang, n=n_tested, correct=accuracy_pct)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(tile(score_color, score_bg, t('explain_score_title', lang), f"{score_pct:.0f}%", score_desc), unsafe_allow_html=True)
    with col2:
        st.markdown(tile("#F59E0B", "#FFFBEB", t('explain_threshold_title', lang), f"{threshold_pct:.0f}%", threshold_desc), unsafe_allow_html=True)
    with col3:
        st.markdown(tile("#8B5CF6", "#F5F3FF", t('explain_accuracy_title', lang), f"{accuracy_pct:.1f}%", accuracy_desc), unsafe_allow_html=True)

    st.markdown(
        f'<div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 15px; border-left: 5px solid #64748B; '
        f'box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;">'
        f'<h4 style="color: #334155; margin-top: 0; font-size: 18px;">{t("explain_note_title", lang)}</h4>'
        f'<p style="color: #475569; font-size: 16px;">{t("explain_note_desc", lang, share=share_pct)}</p>'
        f'<ul style="color: #475569; font-size: 16px; margin-bottom: 0; padding-left: 1.2rem;">'
        f'<li style="margin-bottom: 6px;">{t("explain_recall", lang, recall=recall_pct)}</li>'
        f'<li>{t("explain_precision", lang, precision=precision_pct)}</li>'
        f'</ul></div>',
        unsafe_allow_html=True
    )


# Result Page - Negative
def result_negative_page():
    render_sidebar()
    lang = st.session_state.lang
    
    st.markdown(f"""
    <div style="text-align: left; margin: 1rem 0;">
        <h2 style="background: linear-gradient(90deg, #2E66F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 32px; font-weight: 800; display: inline-block;">{t('report_heading', lang)}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-negative">
        <h1 class="result-title" style="background: linear-gradient(90deg, #05CD99, #10B981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;">{t('neg_title', lang)}</h1>
        <p class="result-subtitle">{t('neg_subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        probability = st.session_state.get('probability', 0)
        diabetes_prob = probability * 100
        non_diabetes_prob = 100 - diabetes_prob
        
        fig = go.Figure(data=[go.Pie(
            labels=[t('chart_non_diab', lang), t('chart_diab', lang)],
            values=[non_diabetes_prob, diabetes_prob],
            hole=.7,
            marker=dict(colors=['#2E66F6', '#E2E8F0']),
            textinfo='none'
        )])
        fig.add_annotation(
            text=f"<b>{int(non_diabetes_prob)}%</b><br><span style='font-size:12px;color:#718096'>{t('chart_safe', lang)}</span>",
            font=dict(size=36, color='#2B3674'), showarrow=False
        )
        fig.update_layout(
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=300, margin=dict(t=50, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown(f"""
        <div class="chart-container" style="height: 100%;">
            <h3 class="chart-title" style="background: linear-gradient(90deg, #F59E0B, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;">{t('analysis_title', lang)}</h3>
            <p class="chart-text">{t('neg_analysis_desc', lang)}</p>
            <div class="highlight-box">{t('neg_highlight', lang)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    render_result_explanation(st.session_state.get('probability', 0), is_positive=False)
    
    input_data = st.session_state.get('input_data', {})
    if input_data:
        render_interactive_insights(input_data)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(t('btn_new_pred', lang), key="new_pred_neg", width="stretch"):
            st.session_state.page = "input"
            st.rerun()

# Result Page - Positive
def result_positive_page():
    render_sidebar()
    lang = st.session_state.lang
    
    st.markdown(f"""
    <div style="text-align: left; margin: 1rem 0;">
        <h2 style="background: linear-gradient(90deg, #2E66F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 32px; font-weight: 800; display: inline-block;">{t('report_heading', lang)}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-positive" style="border: 1px solid #FED7D7;">
        <h1 class="result-title" style="background: linear-gradient(90deg, #E53E3E, #F43F5E); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;">{t('pos_title', lang)}</h1>
        <p class="result-subtitle">{t('pos_subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        probability = st.session_state.get('probability', 0)
        diabetes_prob = probability * 100
        non_diabetes_prob = 100 - diabetes_prob
        
        fig = go.Figure(data=[go.Pie(
            labels=[t('chart_diab', lang), t('chart_non_diab', lang)],
            values=[diabetes_prob, non_diabetes_prob],
            hole=.7,
            marker=dict(colors=['#E53E3E', '#E2E8F0']),
            textinfo='none'
        )])
        fig.add_annotation(
            text=f"<b>{int(diabetes_prob)}%</b><br><span style='font-size:12px;color:#718096'>{t('chart_risk', lang)}</span>",
            font=dict(size=36, color='#2B3674'), showarrow=False
        )
        fig.update_layout(
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=300, margin=dict(t=50, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown(f"""
        <div class="chart-container" style="height: 100%;">
            <h3 class="chart-title" style="background: linear-gradient(90deg, #F59E0B, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block;">{t('analysis_title', lang)}</h3>
            <p class="chart-text">{t('pos_analysis_desc', lang, prob=probability * 100)}</p>
            <div class="highlight-box" style="background: #FFF1F0; color: #C53030; border: 1px solid #FED7D7;">
                {t('pos_highlight', lang)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    render_result_explanation(st.session_state.get('probability', 0), is_positive=True)
    
    input_data = st.session_state.get('input_data', {})
    if input_data:
        render_interactive_insights(input_data)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(t('btn_new_pred', lang), key="new_pred_pos", width="stretch"):
            st.session_state.page = "input"
            st.rerun()

# Main app logic
def main():
    page = st.session_state.page
    if page == "home":
        home_page()
    elif page == "input":
        input_page()
    elif page == "result_negative":
        result_negative_page()
    elif page == "result_positive":
        result_positive_page()

if __name__ == "__main__":
    main()
