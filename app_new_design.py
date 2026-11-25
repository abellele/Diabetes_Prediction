import os
import json
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from translations import t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="SugarSense - Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Design tokens / global CSS (Youcare-inspired dashboard: light panel, white
# rounded cards, royal-blue accent, pill buttons, top navigation bar)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #2F6BFF;
        --primary-dark: #1F55E0;
        --primary-soft: #EAF0FF;
        --ink: #1B2559;
        --muted: #8A94A6;
        --line: #EDF0F5;
        --panel: #F5F7FB;
        --card: #FFFFFF;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}

    .stApp {
        background: #E6EAF1;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Big rounded panel holding the whole dashboard */
    .block-container {
        background: var(--panel);
        border-radius: 32px;
        padding: 1.5rem 2rem 2.5rem 2rem !important;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        max-width: 1320px;
        box-shadow: 0 20px 50px rgba(27, 37, 89, 0.06);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--ink);
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    p, span, div, label, li {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    p, li { color: #64708A; }

    /* ---------- Cards (any container whose key starts with "card") ---------- */
    [class*="st-key-card"] {
        background: var(--card);
        border-radius: 22px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 18px rgba(27, 37, 89, 0.04);
        border: 1px solid var(--line);
    }
    [class*="st-key-cardr1"] {min-height: 300px;}
    [class*="st-key-cardr2"] {min-height: 360px;}

    .card-head {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    .card-title {
        font-size: 16px; font-weight: 700; color: var(--ink) !important; margin: 0;
    }
    .pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px;
        border: 1px solid #D9E2F5; color: var(--primary) !important;
        font-size: 12px; font-weight: 600; background: #fff; white-space: nowrap;
    }
    .pill-dark { background: var(--ink); color: #fff !important; border-color: var(--ink); }
    .pill-soft { background: var(--primary-soft); border-color: var(--primary-soft); }
    .pill-green { background: #E7F8EF; border-color: #E7F8EF; color: #12A150 !important; }
    .pill-red { background: #FDECEC; border-color: #FDECEC; color: #E5484D !important; }

    .info-card {
        background: var(--card);
        padding: 1.5rem 1.6rem;
        border-radius: 22px;
        border: 1px solid var(--line);
        box-shadow: 0 4px 18px rgba(27, 37, 89, 0.04);
        margin: 0.5rem 0 1rem 0;
    }

    /* ---------- Top navigation ---------- */
    .st-key-topnav {
        background: #fff;
        border-radius: 22px;
        padding: 0.65rem 1.1rem;
        border: 1px solid var(--line);
        margin-bottom: 0.75rem;
    }
    .brand {display: flex; align-items: center; gap: 10px; height: 42px;}
    .brand-logo {
        width: 34px; height: 34px; border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #6C9BFF, var(--primary));
        display: flex; align-items: center; justify-content: center;
        color: #fff !important; font-weight: 800; font-size: 16px;
    }
    .brand-name {font-weight: 700; font-size: 18px; color: var(--ink) !important;}
    .nav-icons {display: flex; justify-content: flex-end; align-items: center; gap: 10px; height: 42px;}
    .icon-btn {
        width: 38px; height: 38px; border-radius: 50%; border: 1px solid var(--line);
        display: flex; align-items: center; justify-content: center; background: #fff;
    }
    .avatar {
        width: 40px; height: 40px; border-radius: 50%;
        background: linear-gradient(135deg, #1B2559, #2F6BFF);
        color: #fff !important; font-weight: 700; font-size: 14px;
        display: flex; align-items: center; justify-content: center;
    }

    /* ---------- Buttons ---------- */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 999px;
        font-weight: 600;
        font-size: 14px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    button[data-testid^="stBaseButton-primary"] {
        background: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        color: #fff !important;
        box-shadow: 0 6px 16px rgba(47, 107, 255, 0.28);
    }
    button[data-testid^="stBaseButton-primary"]:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px);
    }
    button[data-testid^="stBaseButton-primary"] p {color: #fff !important;}
    button[data-testid^="stBaseButton-secondary"] {
        background: #F2F4F8 !important;
        border: 1px solid #F2F4F8 !important;
        color: var(--ink) !important;
        box-shadow: none;
    }
    button[data-testid^="stBaseButton-secondary"]:hover {
        border-color: #D9E2F5 !important;
        color: var(--primary) !important;
    }
    button[data-testid^="stBaseButton-secondary"] p {color: inherit !important;}
    button:disabled {opacity: 0.45;}

    /* ---------- Greeting / toolbar ---------- */
    .greeting {font-size: 30px; font-weight: 700; color: var(--ink) !important; margin: 0.4rem 0 0.2rem 0;}
    .greeting span {color: var(--primary) !important;}
    .sub-greeting {color: var(--muted) !important; font-size: 14px; margin: 0;}
    .toolbar {display: flex; gap: 10px; flex-wrap: wrap; margin: 0.9rem 0 1rem 0;}
    .chip {
        display: inline-flex; align-items: center; gap: 8px;
        background: #fff; border: 1px solid var(--line); border-radius: 999px;
        padding: 7px 16px; font-size: 13px; font-weight: 600; color: var(--ink) !important;
    }
    .chip b {color: var(--primary) !important;}

    /* ---------- Blue / Red / Green feature card ---------- */
    .feature-card {
        background: linear-gradient(160deg, #4A82FF 0%, #2F6BFF 55%, #2457E6 100%);
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        min-height: 300px;
        display: flex; flex-direction: column; justify-content: space-between;
        box-shadow: 0 14px 30px rgba(47, 107, 255, 0.30);
        position: relative; overflow: hidden;
    }
    .feature-card.red {
        background: linear-gradient(160deg, #FF7A7A 0%, #E5484D 55%, #C9343A 100%);
        box-shadow: 0 14px 30px rgba(229, 72, 77, 0.28);
    }
    .feature-card.green {
        background: linear-gradient(160deg, #3DD68C 0%, #16A765 55%, #0F8A52 100%);
        box-shadow: 0 14px 30px rgba(22, 167, 101, 0.28);
    }
    .feature-card * {color: #fff !important;}
    .feature-top {display: flex; justify-content: space-between; align-items: center;}
    .feature-dot {
        width: 36px; height: 36px; border-radius: 50%; background: #fff;
        display: flex; align-items: center; justify-content: center;
    }
    .feature-dot::after {content: ''; width: 16px; height: 16px; border-radius: 50%; background: var(--primary);}
    .feature-card.red .feature-dot::after {background: #E5484D;}
    .feature-card.green .feature-dot::after {background: #16A765;}
    .feature-tag {background: #fff; color: var(--primary) !important; border-radius: 999px; padding: 5px 14px; font-size: 12px; font-weight: 600;}
    .feature-card.red .feature-tag {color: #E5484D !important;}
    .feature-card.green .feature-tag {color: #16A765 !important;}
    .feature-title {font-size: 20px; font-weight: 700; margin: 1rem 0 0.5rem 0;}
    .feature-text {font-size: 13px; line-height: 1.6; opacity: 0.92;}
    .feature-bars {display: flex; gap: 6px; margin-top: 1rem;}
    .feature-bars span {flex: 1; height: 4px; border-radius: 4px; background: rgba(255,255,255,0.35);}
    .feature-bars span.on {background: #fff;}

    /* ---------- Lists ---------- */
    .list-row {
        display: flex; align-items: center; gap: 12px;
        padding: 0.7rem 0; border-bottom: 1px solid var(--line);
    }
    .list-row:last-child {border-bottom: none;}
    .list-icon {
        width: 40px; height: 40px; border-radius: 50%; background: var(--primary-soft);
        display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
    }
    .list-main {font-size: 14px; font-weight: 700; color: var(--ink) !important; margin: 0;}
    .list-sub {font-size: 12px; color: var(--muted) !important; margin: 0;}
    .progress {height: 6px; background: #EDF1F8; border-radius: 6px; overflow: hidden; margin-top: 6px;}
    .progress > div {height: 100%; border-radius: 6px;}

    .kv-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 1rem 1.2rem; margin-top: 0.8rem;}
    .kv-label {font-size: 12px; color: var(--muted) !important; margin: 0;}
    .kv-value {font-size: 14px; font-weight: 700; color: var(--ink) !important; margin: 2px 0 0 0;}

    .big-num {font-size: 34px; font-weight: 800; color: var(--ink) !important; line-height: 1;}

    /* ---------- Inputs ---------- */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: #F7F9FC !important;
        color: var(--ink) !important;
        font-weight: 500;
    }
    .stNumberInput label p, .stSelectbox label p, .stSlider label p {
        color: var(--ink) !important; font-weight: 600; font-size: 13px;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {gap: 8px; border-bottom: none;}
    .stTabs [data-baseweb="tab"] {
        background: #F2F4F8; border-radius: 999px; padding: 6px 18px; height: auto;
    }
    .stTabs [data-baseweb="tab"] p {font-weight: 600; color: var(--ink) !important;}
    .stTabs [aria-selected="true"] {background: var(--primary) !important;}
    .stTabs [aria-selected="true"] p {color: #fff !important;}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

    .disclaimer-mini {
        background: #FFF7ED; border: 1px solid #FFE4C7; border-radius: 14px;
        padding: 0.8rem 1rem; font-size: 12px; color: #B45309 !important; margin-top: 0.8rem;
    }

    @media (max-width: 768px) {
        .block-container {padding: 1rem !important; border-radius: 20px;}
        .greeting {font-size: 24px;}
    }
</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {"displayModeBar": False}

# ---------------------------------------------------------------------------
# Model and Feature Configuration (all 14 input features)
# Encodings: gender (Male=0, Female=1), height in meters, yes/no fields as 0/1.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'age', 'gender', 'pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height',
    'weight', 'bmi', 'family_diabetes', 'hypertensive', 'family_hypertension',
    'cardiovascular_disease', 'stroke'
]
OPTIMAL_THRESHOLD = 0.72  # Validation-derived optimal threshold for Ensemble Stacking (14 features)

FALLBACK_MEAN = np.array([
    45.8563, 0.7098, 76.6386, 134.1616, 82.2852, 7.2629, 1.5502,
    53.5635, 22.2454, 0.0327, 0.1105, 0.0351, 0.0105, 0.0041
])
FALLBACK_SCALE = np.array([
    13.3576, 0.4539, 11.8925, 21.5387, 12.2587, 1.8129, 0.0698,
    9.7586, 3.7197, 0.1778, 0.3135, 0.1841, 0.1021, 0.0635
])


def html(markup):
    """Render HTML safely through st.markdown (strips indentation/blank lines so
    Markdown never turns it into a code block)."""
    clean = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


def card_header(title, pill=None, pill_class=""):
    pill_html = f'<span class="pill {pill_class}">{pill}</span>' if pill else ""
    html(f'<div class="card-head"><p class="card-title">{title}</p>{pill_html}</div>')


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


@st.cache_data
def load_dataset():
    data_path = os.path.join(BASE_DIR, "dataset.csv")
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            df["is_diabetic"] = (df["diabetic"].astype(str).str.strip().str.lower() == "yes").astype(int)
            return df
        except Exception:
            return None
    return None


model, scaler, optimal_threshold, model_metrics = load_model_and_scaler()
dataset = load_dataset()


def scale_input_features(raw_dict):
    df_in = pd.DataFrame([raw_dict])[FEATURE_NAMES]
    if scaler is not None:
        scaled_vals = scaler.transform(df_in)
    else:
        scaled_vals = (df_in.values - FALLBACK_MEAN) / FALLBACK_SCALE
    return pd.DataFrame(scaled_vals, columns=FEATURE_NAMES)


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'lang' not in st.session_state:
    st.session_state.lang = "en"
if 'history' not in st.session_state:
    st.session_state.history = []


def go_to(page):
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Top navigation bar (replaces sidebar)
# ---------------------------------------------------------------------------
BELL_SVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1B2559" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>'


def render_navbar():
    page = st.session_state.page
    lang = st.session_state.lang
    has_result = 'prediction' in st.session_state
    with st.container(key="topnav"):
        c_brand, c_gap1, c_d, c_p, c_r, c_gap2, c_lang, c_icons = st.columns(
            [2.1, 0.8, 1.15, 1.15, 1.15, 0.8, 1.15, 1.2], vertical_alignment="center")
        with c_brand:
            html('<div class="brand"><div class="brand-logo">S</div><span class="brand-name">SugarSense</span></div>')
        with c_d:
            if st.button(t('nav_dashboard', lang), key="nav_home", type="primary" if page == "home" else "secondary"):
                go_to("home")
        with c_p:
            if st.button(t('nav_prediction', lang), key="nav_input", type="primary" if page == "input" else "secondary"):
                go_to("input")
        with c_r:
            is_report = page in ("result_positive", "result_negative")
            if st.button(t('nav_report', lang), key="nav_report", type="primary" if is_report else "secondary",
                         disabled=not has_result):
                go_to("result_positive" if st.session_state.prediction == 1 else "result_negative")
        with c_lang:
            lang_btn_label = "🌐 ID" if lang == "en" else "🌐 EN"
            if st.button(lang_btn_label, key="nav_lang_toggle", type="secondary", help=t('lang_label', lang)):
                st.session_state.lang = "id" if lang == "en" else "en"
                st.rerun()
        with c_icons:
            html(f'<div class="nav-icons"><div class="icon-btn">{BELL_SVG}</div>'
                 f'<div class="avatar">SS</div></div>')


def greeting_text(lang):
    hour = datetime.now().hour
    if hour < 12:
        return t('greet_morning', lang)
    if hour < 18:
        return t('greet_afternoon', lang)
    return t('greet_evening', lang)


def render_page_header(title_html, subtitle, button_label=None, button_key=None, target=None):
    c_left, c_right = st.columns([4, 1], vertical_alignment="center")
    with c_left:
        html(f'<p class="greeting">{title_html}</p><p class="sub-greeting">{subtitle}</p>')
    with c_right:
        if button_label and st.button(button_label, key=button_key, type="primary"):
            go_to(target)


# ---------------------------------------------------------------------------
# Dashboard charts (built from dataset.csv)
# ---------------------------------------------------------------------------
def base_layout(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#8A94A6", size=11),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1B2559", font_color="#fff", bordercolor="#1B2559"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F6", zeroline=False, showline=False, ticks="")
    return fig


AGE_BINS = [0, 30, 40, 50, 60, 70, 200]
AGE_LABELS = ["<30", "30s", "40s", "50s", "60s", "70+"]


def glucose_by_age_chart(df, lang):
    d = df.copy()
    d["age_group"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    g = d.groupby(["age_group", "is_diabetic"], observed=True)["glucose"].mean().unstack()
    fig = go.Figure()
    if 1 in g.columns:
        fig.add_trace(go.Scatter(x=g.index.astype(str), y=g[1].round(2), mode="lines",
                                 name=t('chart_diab', lang),
                                 line=dict(color="#2F6BFF", width=3, shape="spline")))
    if 0 in g.columns:
        fig.add_trace(go.Scatter(x=g.index.astype(str), y=g[0].round(2), mode="lines",
                                 name=t('chart_non_diab', lang),
                                 line=dict(color="#A9C1FF", width=2.5, shape="spline")))
    return base_layout(fig, 190)


def prevalence_by_age_chart(df):
    d = df.copy()
    d["age_group"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    r = d.groupby("age_group", observed=True)["is_diabetic"].mean() * 100
    fig = go.Figure(go.Scatter(
        x=r.index.astype(str), y=r.round(1), mode="lines+markers", fill="tozeroy",
        line=dict(color="#2F6BFF", width=2.5), marker=dict(size=6, color="#2F6BFF"),
        fillcolor="rgba(47,107,255,0.12)", hovertemplate="%{x}: %{y}%<extra></extra>"))
    fig = base_layout(fig, 170)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig


def bmi_risk_chart(df, lang):
    d = df.copy()
    d["bmi_cat"] = pd.cut(d["bmi"], [0, 18.5, 25, 30, 35, 100],
                          labels=["Under", "Normal", "Over", "Obese I", "Obese II+"], right=False)
    r = d.groupby("bmi_cat", observed=True)["is_diabetic"].mean() * 100
    overall = d["is_diabetic"].mean() * 100
    colors = ["#2F6BFF" if v >= overall else "#E3EBFF" for v in r.values]
    fig = go.Figure(go.Bar(
        x=r.index.astype(str), y=r.round(1),
        marker=dict(color=colors, cornerradius=10, line=dict(color="#C9D8FF", width=1)),
        width=0.55, hovertemplate="BMI %{x}: %{y}%<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color="#1B2559", line_width=1.5,
                  annotation_text=f"{t('avg_label', lang)} {overall:.1f}%",
                  annotation_position="top left",
                  annotation_font=dict(color="#1B2559", size=11))
    fig = base_layout(fig, 250)
    fig.update_yaxes(ticksuffix="%")
    return fig


# ---------------------------------------------------------------------------
# Home / Dashboard page
# ---------------------------------------------------------------------------
def home_page():
    render_navbar()
    lang = st.session_state.lang
    render_page_header(f"{greeting_text(lang)}, <span>{t('greet_name', lang)}</span>",
                       t('hero_subtitle', lang),
                       t('btn_check_new', lang), "hero_check", "input")

    n_rows = len(dataset) if dataset is not None else 0
    prevalence = dataset["is_diabetic"].mean() * 100 if dataset is not None else 0
    acc_val = model_metrics.get('test_accuracy', 0) * 100
    acc_chip = f'<span class="chip">✅ {t("chip_accuracy", lang)} <b>{acc_val:.1f}%</b></span>' if acc_val else ""

    html(f"""
    <div class="toolbar">
        <span class="chip">🧪 {t('chip_features', lang)} <b>{len(FEATURE_NAMES)}</b></span>
        <span class="chip">🎯 {t('chip_threshold', lang)} <b>{optimal_threshold * 100:.0f}%</b></span>
        {acc_chip}
        <span class="chip">🗂️ {t('chip_dataset', lang)} <b>{n_rows:,} {t('chip_records', lang)}</b></span>
        <span class="chip">📅 {datetime.now().strftime('%d %B %Y')}</span>
    </div>
    """)

    # ---------------- Row 1 ----------------
    c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1])
    with c1:
        with st.container(key="cardr1_glucose"):
            card_header(t('card_glucose_trend', lang), t('pill_dataset', lang))
            html(f'<div style="display:flex; gap:8px; margin-bottom:4px;">'
                 f'<span class="pill pill-dark">● {t("chart_diab", lang)}</span>'
                 f'<span class="pill pill-soft">● {t("chart_non_diab", lang)}</span></div>')
            if dataset is not None:
                st.plotly_chart(glucose_by_age_chart(dataset, lang), width="stretch", config=PLOT_CONFIG)
            else:
                st.info("dataset.csv not found.")

    with c2:
        html(f"""
        <div class="feature-card">
            <div>
                <div class="feature-top"><div class="feature-dot"></div><span class="feature-tag">{t('pill_today', lang)}</span></div>
                <p class="feature-title">{t('what_is_title', lang)}</p>
                <p class="feature-text">{t('what_is_desc', lang)}</p>
            </div>
            <div class="feature-bars"><span class="on"></span><span></span><span></span></div>
        </div>
        """)

    with c3:
        with st.container(key="cardr1_prev"):
            card_header(t('card_prevalence', lang), "···")
            html(f'<div style="display:flex; align-items:center; gap:10px;"><span class="big-num">{prevalence:.1f}%</span>'
                 f'<span class="pill pill-soft">{t("of_people", lang, n=n_rows)}</span></div>')
            if dataset is not None:
                st.plotly_chart(prevalence_by_age_chart(dataset), width="stretch", config=PLOT_CONFIG)

    with c4:
        with st.container(key="cardr1_history"):
            card_header(t('card_recent', lang), "···")
            history = st.session_state.history[-3:][::-1]
            if not history:
                html(f'<div class="list-row"><div class="list-icon">🩺</div><div>'
                     f'<p class="list-main">{t("no_checks_title", lang)}</p>'
                     f'<p class="list-sub">{t("no_checks_sub", lang)}</p></div></div>')
            for item in history:
                risk = item["probability"] * 100
                color = "#E5484D" if item["prediction"] == 1 else "#2F6BFF"
                label = t('chart_diab', lang) if item["prediction"] == 1 else t('chart_non_diab', lang)
                html(f"""
                <div class="list-row">
                    <div class="list-icon">📋</div>
                    <div style="flex:1;">
                        <p class="list-main">{item['time']}</p>
                        <p class="list-sub">{label} · {risk:.1f}%</p>
                        <div class="progress"><div style="width:{max(risk, 3):.0f}%; background:{color};"></div></div>
                    </div>
                </div>
                """)

    # ---------------- Row 2 ----------------
    d1, d2, d3 = st.columns([1, 1.9, 1])
    with d1:
        with st.container(key="cardr2_info"):
            last = st.session_state.get('input_data')
            card_header(t('card_health_info', lang), t('pill_last_input', lang) if last else t('pill_model', lang))
            if last:
                html(f"""
                <div class="list-row" style="border-bottom:1px solid #EDF0F5;">
                    <div class="avatar">You</div>
                    <div><p class="list-main">{t('latest_check', lang)}</p><p class="list-sub">{st.session_state.history[-1]['time'] if st.session_state.history else ''}</p></div>
                </div>
                <div class="kv-grid">
                    <div><p class="kv-label">{t('range_glucose', lang)}</p><p class="kv-value">{last['glucose']} mmol/L</p></div>
                    <div><p class="kv-label">BMI</p><p class="kv-value">{last['bmi']}</p></div>
                    <div><p class="kv-label">{t('range_bp', lang)}</p><p class="kv-value">{last['systolic_bp']:.0f}/{last['diastolic_bp']:.0f} mmHg</p></div>
                    <div><p class="kv-label">{t('range_pulse', lang)}</p><p class="kv-value">{last['pulse_rate']:.0f} bpm</p></div>
                    <div><p class="kv-label">{t('label_age', lang)}</p><p class="kv-value">{last['age']}</p></div>
                    <div><p class="kv-label">{t('label_hypertensive', lang)}</p><p class="kv-value">{t('opt_yes', lang) if last['hypertensive'] == 1 else t('opt_no', lang)}</p></div>
                </div>
                """)
            else:
                acc_str = f"{acc_val:.1f}%" if acc_val else "-"
                html(f"""
                <p style="font-size:13px; line-height:1.6;">{t('what_do_desc', lang)}</p>
                <div class="kv-grid">
                    <div><p class="kv-label">{t('kv_algorithm', lang)}</p><p class="kv-value">Stacking Ensemble</p></div>
                    <div><p class="kv-label">{t('kv_features', lang)}</p><p class="kv-value">{t('kv_features_val', lang)}</p></div>
                    <div><p class="kv-label">{t('kv_threshold', lang)}</p><p class="kv-value">{optimal_threshold * 100:.0f}%</p></div>
                    <div><p class="kv-label">{t('kv_accuracy', lang)}</p><p class="kv-value">{acc_str}</p></div>
                    <div><p class="kv-label">{t('kv_training', lang)}</p><p class="kv-value">{n_rows:,} {t('chip_records', lang)}</p></div>
                    <div><p class="kv-label">{t('kv_output', lang)}</p><p class="kv-value">{t('chart_diab', lang)} / {t('chart_non_diab', lang)}</p></div>
                </div>
                """)

    with d2:
        with st.container(key="cardr2_bmi"):
            card_header(t('card_bmi_rate', lang), t('pill_dataset', lang))
            html(f'<div style="display:flex; gap:14px; justify-content:flex-end; font-size:12px;">'
                 f'<span style="color:#8A94A6;">▢ {t("below_avg", lang)}</span>'
                 f'<span style="color:#2F6BFF;">■ {t("above_avg", lang)}</span></div>')
            if dataset is not None:
                st.plotly_chart(bmi_risk_chart(dataset, lang), width="stretch", config=PLOT_CONFIG)

    with d3:
        with st.container(key="cardr2_ranges"):
            card_header(t('card_ranges', lang), t('pill_guide', lang))
            for icon, name, rng in [
                ("🩸", t('range_glucose', lang), "4.0 – 7.0 mmol/L"),
                ("⚖️", t('range_bmi', lang), "18.5 – 24.9"),
                ("🫀", t('range_bp', lang), "< 120 / 80 mmHg"),
                ("❤️", t('range_pulse', lang), "60 – 100 bpm"),
            ]:
                html(f'<div class="list-row"><div class="list-icon">{icon}</div><div>'
                     f'<p class="list-main">{name}</p><p class="list-sub">{rng}</p></div></div>')
            html(f'<div class="disclaimer-mini">{t("disclaimer_title", lang)} — {t("disclaimer_desc", lang)}</div>')


# ---------------------------------------------------------------------------
# Input page (14 features)
# ---------------------------------------------------------------------------
def input_page():
    render_navbar()
    lang = st.session_state.lang
    render_page_header(f'<span>{t("form_title", lang)}</span>', t('form_subtitle', lang))
    html(f'<div class="toolbar"><span class="chip"><b>{t("step_1", lang)}</b></span>'
         f'<span class="chip">{t("step_2", lang)}</span><span class="chip">{t("step_3", lang)}</span></div>')

    col_form, col_side = st.columns([2.2, 1])

    with col_form:
        with st.container(key="card_form"):
            card_header(t('card_your_metrics', lang), t('pill_fields', lang))
            with st.form("prediction_form", border=False):
                col1, col2 = st.columns([1, 1])
                yes_no = lambda x: t('opt_no', lang) if x == 0 else t('opt_yes', lang)

                with col1:
                    age = st.number_input(
                        t('label_age', lang),
                        min_value=0,
                        max_value=120,
                        value=35,
                        step=1,
                        help=t('help_age', lang)
                    )

                    gender = st.selectbox(
                        t('label_gender', lang),
                        options=[0, 1],
                        format_func=lambda x: t('opt_male', lang) if x == 0 else t('opt_female', lang)
                    )

                    height_cm = st.number_input(
                        t('label_height', lang),
                        min_value=0.0,
                        value=160.0,
                        step=1.0,
                        help=t('help_height', lang)
                    )

                    weight = st.number_input(
                        t('label_weight', lang),
                        min_value=0.0,
                        value=60.0,
                        step=0.1
                    )

                    bmi = st.number_input(
                        t('label_bmi', lang),
                        min_value=0.0,
                        value=22.0,
                        step=0.1,
                        help=t('help_bmi', lang)
                    )

                    glucose = st.number_input(
                        t('label_glucose', lang),
                        min_value=0.0,
                        value=5.0,
                        step=0.1,
                        help=t('help_glucose', lang)
                    )

                    pulse_rate = st.number_input(
                        t('label_pulse', lang),
                        min_value=0.0,
                        value=70.0,
                        step=1.0,
                        help=t('help_pulse', lang)
                    )

                with col2:
                    systolic_bp = st.number_input(
                        t('label_systolic', lang),
                        min_value=0.0,
                        value=120.0,
                        step=1.0,
                        help=t('help_systolic', lang)
                    )

                    diastolic_bp = st.number_input(
                        t('label_diastolic', lang),
                        min_value=0.0,
                        value=80.0,
                        step=1.0,
                        help=t('help_diastolic', lang)
                    )

                    hypertensive = st.selectbox(
                        t('label_hypertensive', lang),
                        options=[0, 1],
                        format_func=yes_no
                    )

                    family_hypertension = st.selectbox(
                        t('label_family_hypertension', lang),
                        options=[0, 1],
                        format_func=yes_no,
                        help=t('help_family_hypertension', lang)
                    )

                    family_diabetes = st.selectbox(
                        t('label_family_diabetes', lang),
                        options=[0, 1],
                        format_func=yes_no,
                        help=t('help_family_diabetes', lang)
                    )

                    cardiovascular_disease = st.selectbox(
                        t('label_cardiovascular', lang),
                        options=[0, 1],
                        format_func=yes_no,
                        help=t('help_cardiovascular', lang)
                    )

                    stroke = st.selectbox(
                        t('label_stroke', lang),
                        options=[0, 1],
                        format_func=yes_no,
                        help=t('help_stroke', lang)
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    submit = st.form_submit_button(t('btn_predict', lang), type="primary", width="stretch")

    with col_side:
        html(f"""
        <div class="feature-card" style="min-height: 250px;">
            <div>
                <div class="feature-top"><div class="feature-dot"></div><span class="feature-tag">{t('pill_tips', lang)}</span></div>
                <p class="feature-title">{t('tips_title', lang)}</p>
                <p class="feature-text">{t('tips_desc', lang)}</p>
            </div>
            <div class="feature-bars"><span></span><span class="on"></span><span></span></div>
        </div>
        """)
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        with st.container(key="card_ranges_input"):
            card_header(t('card_ranges', lang), t('pill_guide', lang))
            for icon, name, rng in [
                ("🩸", t('range_glucose', lang), "4.0 – 7.0 mmol/L"),
                ("⚖️", "BMI", "18.5 – 24.9"),
                ("🫀", t('range_bp', lang), "< 120 / 80 mmHg"),
                ("❤️", t('range_pulse', lang), "60 – 100 bpm"),
            ]:
                html(f'<div class="list-row"><div class="list-icon">{icon}</div><div>'
                     f'<p class="list-main">{name}</p><p class="list-sub">{rng}</p></div></div>')

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
                st.session_state.history.append({
                    "time": datetime.now().strftime("%d %b %Y, %H:%M"),
                    "probability": probability,
                    "prediction": prediction,
                })

                go_to("result_positive" if prediction == 1 else "result_negative")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")


# ---------------------------------------------------------------------------
# Interactive Health Insights (Deep Dive -> What-If Simulator -> Visual Gauges)
# ---------------------------------------------------------------------------
def render_interactive_insights(input_data):
    lang = st.session_state.lang
    card_header(t('insights_title', lang), t('pill_personalized', lang))

    # Tab order: Deep Dive, What-If Simulator, Visual Gauges
    tab_deep_dive, tab_simulator, tab_gauges = st.tabs([
        t('tab_deep_dive', lang),
        t('tab_simulator', lang),
        t('tab_gauges', lang)
    ])

    with tab_deep_dive:
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        # BMI
        bmi = input_data.get('bmi', 0)
        if bmi < 18.5:
            bmi_color, bmi_bg = "#F59E0B", "#FFFBEB"
            bmi_text, bmi_stat = t('plan_underweight', lang), t('stat_underweight', lang)
        elif 25.0 <= bmi < 30.0:
            bmi_color, bmi_bg = "#F59E0B", "#FFFBEB"
            bmi_text, bmi_stat = t('plan_overweight', lang), t('stat_overweight', lang)
        elif bmi >= 30.0:
            bmi_color, bmi_bg = "#EF4444", "#FEF2F2"
            bmi_text, bmi_stat = t('plan_obese', lang), t('stat_obese', lang)
        else:
            bmi_color, bmi_bg = "#10B981", "#ECFDF5"
            bmi_text, bmi_stat = t('plan_healthy_weight', lang), t('stat_healthy_weight', lang)

        with col_a:
            html(f"""
            <div style="background-color: {bmi_bg}; padding: 1.4rem; border-radius: 16px; border-left: 5px solid {bmi_color}; height: 100%; margin-bottom: 1rem;">
                <h4 style="color: {bmi_color}; margin-top: 0; font-size: 17px;">{t('card_bmi_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 8px 0;">{bmi} <span style="font-size: 16px; font-weight: 600; color: {bmi_color};">({bmi_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {bmi_text}</p>
            </div>
            """)

        # Blood Sugar
        glucose = input_data.get('glucose', 0)
        if glucose > 7.0:
            gl_color, gl_bg = "#F59E0B", "#FFFBEB"
            gl_text, gl_stat = t('plan_above_normal', lang), t('stat_above_normal', lang)
        elif glucose < 4.0:
            gl_color, gl_bg = "#F59E0B", "#FFFBEB"
            gl_text, gl_stat = t('plan_below_normal', lang), t('stat_below_normal', lang)
        else:
            gl_color, gl_bg = "#10B981", "#ECFDF5"
            gl_text, gl_stat = t('plan_normal_glucose', lang), t('stat_normal_glucose', lang)

        with col_b:
            html(f"""
            <div style="background-color: {gl_bg}; padding: 1.4rem; border-radius: 16px; border-left: 5px solid {gl_color}; height: 100%; margin-bottom: 1rem;">
                <h4 style="color: {gl_color}; margin-top: 0; font-size: 17px;">{t('card_glucose_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 8px 0;">{glucose} <span style="font-size: 16px; font-weight: 600; color: {gl_color};">({gl_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {gl_text}</p>
            </div>
            """)

        col_c, col_d = st.columns(2)

        # Blood Pressure
        systolic = input_data.get('systolic_bp', 0)
        diastolic = input_data.get('diastolic_bp', 0)
        hypertensive = input_data.get('hypertensive', 0)
        if systolic >= 130 or diastolic >= 80 or hypertensive == 1:
            bp_color, bp_bg = "#EF4444", "#FEF2F2"
            bp_text, bp_stat = t('plan_elevated_bp', lang), t('stat_elevated_bp', lang)
        else:
            bp_color, bp_bg = "#10B981", "#ECFDF5"
            bp_text, bp_stat = t('plan_normal_bp', lang), t('stat_normal_bp', lang)

        with col_c:
            html(f"""
            <div style="background-color: {bp_bg}; padding: 1.4rem; border-radius: 16px; border-left: 5px solid {bp_color}; height: 100%; margin-bottom: 1rem;">
                <h4 style="color: {bp_color}; margin-top: 0; font-size: 17px;">{t('card_bp_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 8px 0;">{systolic:.0f}/{diastolic:.0f} <span style="font-size: 16px; font-weight: 600; color: {bp_color};">({bp_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {bp_text}</p>
            </div>
            """)

        # Heart Rate
        pulse = input_data.get('pulse_rate', 0)
        if pulse > 100:
            hr_color, hr_bg = "#F59E0B", "#FFFBEB"
            hr_text, hr_stat = t('plan_high_hr', lang), t('stat_high_hr', lang)
        elif 0 < pulse < 60:
            hr_color, hr_bg = "#8B5CF6", "#F5F3FF"
            hr_text, hr_stat = t('plan_low_hr', lang), t('stat_low_hr', lang)
        else:
            hr_color, hr_bg = "#10B981", "#ECFDF5"
            hr_text, hr_stat = t('plan_normal_hr', lang), t('stat_normal_hr', lang)

        with col_d:
            html(f"""
            <div style="background-color: {hr_bg}; padding: 1.4rem; border-radius: 16px; border-left: 5px solid {hr_color}; height: 100%; margin-bottom: 1rem;">
                <h4 style="color: {hr_color}; margin-top: 0; font-size: 17px;">{t('card_hr_title', lang)}</h4>
                <p style="font-size: 24px; font-weight: 800; color: #1E293B; margin: 8px 0;">{pulse:.0f} <span style="font-size: 16px; font-weight: 600; color: {hr_color};">({hr_stat})</span></p>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;"><strong style="font-size: 16px; font-weight: 600;">{t('action_plan', lang)}</strong> {hr_text}</p>
            </div>
            """)

    with tab_simulator:
        html(f"<p style='color: #64708A; margin: 1rem 0;'>{t('sim_desc', lang)}</p>")

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
                'glucose': sim_glucose,
                'systolic_bp': sim_sys_bp,
                'diastolic_bp': sim_dia_bp,
                'bmi': sim_bmi,
                'weight': sim_weight,
                'pulse_rate': sim_pulse
            })
            sim_scaled_df = scale_input_features(sim_input)
            sim_prob = float(model.predict_proba(sim_scaled_df)[0][1]) * 100
            threshold_pct = optimal_threshold * 100
            is_above_thresh = sim_prob >= threshold_pct
            sim_msg = t('sim_above', lang, thresh=threshold_pct) if is_above_thresh else t('sim_below', lang, thresh=threshold_pct)

            html(f"""
            <div style="background: {'#FFF1F0' if is_above_thresh else '#F0FFF4'}; padding: 1.4rem; border-radius: 16px; text-align: center; margin-top: 1rem; border: 1px solid {'#FED7D7' if is_above_thresh else '#C6F6D5'};">
                <h4 style="color: {'#E53E3E' if is_above_thresh else '#16A765'}; margin: 0 0 0.4rem 0;">{t('sim_title', lang)}</h4>
                <h2 style="color: {'#E53E3E' if is_above_thresh else '#16A765'}; font-size: 32px; font-weight: 800; margin: 0;">{sim_prob:.1f}%</h2>
                <p style="color: #64708A; margin: 0.5rem 0 0 0;">{sim_msg}</p>
            </div>
            """)
        else:
            st.error(t('sim_error', lang))

    with tab_gauges:
        st.markdown("<br>", unsafe_allow_html=True)

        def create_gauge(value, title, min_val, max_val, steps):
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title, 'font': {'size': 15, 'color': '#1B2559'}},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': "#2F6BFF", 'thickness': 0.2},
                    'steps': steps
                }
            ))
            val_text = f"{value:.1f}" if isinstance(value, (float, int)) and value % 1 != 0 else f"{int(value)}"
            fig.add_annotation(
                x=0.5, y=0.15, xref="paper", yref="paper",
                text=val_text, font=dict(size=30, color='#1B2559'), showarrow=False
            )
            fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
            return fig

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            bmi_steps = [
                {'range': [10, 18.5], 'color': '#FEEBC8'},
                {'range': [18.5, 25], 'color': '#C6F6D5'},
                {'range': [25, 30], 'color': '#FEEBC8'},
                {'range': [30, 45], 'color': '#FED7D7'}
            ]
            st.plotly_chart(create_gauge(input_data.get('bmi', 0), "BMI", 10, 45, bmi_steps),
                            width="stretch", config=PLOT_CONFIG)

            bp_steps = [
                {'range': [60, 120], 'color': '#C6F6D5'},
                {'range': [120, 130], 'color': '#FEEBC8'},
                {'range': [130, 200], 'color': '#FED7D7'}
            ]
            st.plotly_chart(create_gauge(input_data.get('systolic_bp', 0), t('label_systolic', lang), 60, 200, bp_steps),
                            width="stretch", config=PLOT_CONFIG)

        with col_g2:
            glucose_steps = [
                {'range': [0, 4.0], 'color': '#FEEBC8'},
                {'range': [4.0, 7.0], 'color': '#C6F6D5'},
                {'range': [7.0, 15.0], 'color': '#FED7D7'}
            ]
            st.plotly_chart(create_gauge(input_data.get('glucose', 0), t('label_glucose', lang), 0, 15, glucose_steps),
                            width="stretch", config=PLOT_CONFIG)

            pulse_steps = [
                {'range': [40, 60], 'color': '#FEEBC8'},
                {'range': [60, 100], 'color': '#C6F6D5'},
                {'range': [100, 160], 'color': '#FED7D7'}
            ]
            st.plotly_chart(create_gauge(input_data.get('pulse_rate', 0), t('label_pulse', lang), 40, 160, pulse_steps),
                            width="stretch", config=PLOT_CONFIG)


# ---------------------------------------------------------------------------
# How to Read Your Result (simple & easy-to-understand explanation)
# ---------------------------------------------------------------------------
def render_result_explanation(probability, is_positive):
    """Show the risk score, decision threshold and model accuracy in plain, short sentences."""
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

    score_desc = t('explain_score_desc', lang)
    if not is_positive:
        score_desc += " " + t('explain_safe_note', lang)
    score_color, score_bg = ("#E5484D", "#FEF2F2") if is_positive else ("#12A150", "#ECFDF5")

    verdict_key = 'pos_title' if is_positive else 'neg_title'
    compare_key = 'explain_threshold_above' if is_positive else 'explain_threshold_below'
    threshold_desc = (
        t('explain_threshold_desc', lang) + " "
        + t(compare_key, lang, score=score_pct, label=t(verdict_key, lang))
    )
    accuracy_desc = t('explain_accuracy_desc', lang, n=n_tested, correct=accuracy_pct)

    html(f"""
    <div class="info-card">
        <div class="card-head">
            <p class="card-title">{t('explain_title', lang)}</p>
            <span class="pill">{t('pill_how_to_read', lang)}</span>
        </div>
        <p style="font-size: 13px; color: #8A94A6; margin: 0 0 1rem 0;">{t('explain_intro', lang)}</p>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="flex: 1; min-width: 200px; background: {score_bg}; border-left: 5px solid {score_color}; padding: 1.1rem 1.25rem; border-radius: 16px;">
                <p style="margin: 0; font-size: 13px; font-weight: 600; color: {score_color};">{t('explain_score_title', lang)}</p>
                <p style="margin: 6px 0; font-size: 28px; font-weight: 800; color: #1B2559;">{score_pct:.0f}%</p>
                <p style="margin: 0; font-size: 13px; color: #475569; line-height: 1.5;">{score_desc}</p>
            </div>
            <div style="flex: 1; min-width: 200px; background: #FFFBEB; border-left: 5px solid #F59E0B; padding: 1.1rem 1.25rem; border-radius: 16px;">
                <p style="margin: 0; font-size: 13px; font-weight: 600; color: #D97706;">{t('explain_threshold_title', lang)}</p>
                <p style="margin: 6px 0; font-size: 28px; font-weight: 800; color: #1B2559;">{threshold_pct:.0f}%</p>
                <p style="margin: 0; font-size: 13px; color: #475569; line-height: 1.5;">{threshold_desc}</p>
            </div>
            <div style="flex: 1; min-width: 200px; background: #F4F7FE; border-left: 5px solid #2F6BFF; padding: 1.1rem 1.25rem; border-radius: 16px;">
                <p style="margin: 0; font-size: 13px; font-weight: 600; color: #2F6BFF;">{t('explain_accuracy_title', lang)}</p>
                <p style="margin: 6px 0; font-size: 28px; font-weight: 800; color: #1B2559;">{accuracy_pct:.1f}%</p>
                <p style="margin: 0; font-size: 13px; color: #475569; line-height: 1.5;">{accuracy_desc}</p>
            </div>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #EDF0F5; padding: 1rem 1.25rem; border-radius: 14px;">
            <p style="margin: 0 0 0.4rem 0; font-size: 14px; font-weight: 700; color: #1B2559;">{t('explain_note_title', lang)}</p>
            <p style="margin: 0 0 0.5rem 0; font-size: 13px; color: #475569;">{t('explain_note_desc', lang, share=share_pct)}</p>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 13px; color: #475569; line-height: 1.65;">
                <li>{t('explain_recall', lang, recall=recall_pct)}</li>
                <li>{t('explain_precision', lang, precision=precision_pct)}</li>
            </ul>
        </div>
    </div>
    """)


# ---------------------------------------------------------------------------
# Result page (shared by positive & negative outcomes)
# ---------------------------------------------------------------------------
def donut_chart(probability, positive, lang):
    diabetes_prob = probability * 100
    non_diabetes_prob = 100 - diabetes_prob
    if positive:
        labels = [t('chart_diab', lang), t('chart_non_diab', lang)]
        values = [diabetes_prob, non_diabetes_prob]
        colors, center, caption = ['#E5484D', '#EDF1F8'], diabetes_prob, t('chart_risk', lang)
    else:
        labels = [t('chart_non_diab', lang), t('chart_diab', lang)]
        values = [non_diabetes_prob, diabetes_prob]
        colors, center, caption = ['#2F6BFF', '#EDF1F8'], non_diabetes_prob, t('chart_safe', lang)

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.74, sort=False,
        marker=dict(colors=colors), textinfo='none',
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
    )])
    fig.add_annotation(
        text=f"<b>{center:.0f}%</b><br><span style='font-size:13px;color:#8A94A6'>{caption}</span>",
        font=dict(size=34, color='#1B2559'), showarrow=False
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                    font=dict(color="#64708A", size=12)),
        height=250, margin=dict(t=5, b=5, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans, sans-serif"),
    )
    return fig


def result_page(positive):
    render_navbar()
    lang = st.session_state.lang
    probability = st.session_state.get('probability', 0)
    diabetes_pct = probability * 100
    acc_val = model_metrics.get('test_accuracy', 0) * 100
    acc_chip = f'<span class="chip">✅ {t("chip_accuracy", lang)} <b>{acc_val:.1f}%</b></span>' if acc_val else ""

    render_page_header(f'<span>{t("report_heading", lang)}</span>',
                       t('report_subtitle', lang),
                       f"+  {t('btn_new_pred', lang)}", "new_pred_top", "input")

    html(f"""
    <div class="toolbar">
        <span class="chip">📅 {st.session_state.history[-1]['time'] if st.session_state.history else datetime.now().strftime('%d %b %Y')}</span>
        <span class="chip">🩸 {t('chip_risk', lang)} <b>{diabetes_pct:.1f}%</b></span>
        <span class="chip">🎯 {t('chip_threshold', lang)} <b>{optimal_threshold * 100:.0f}%</b></span>
        {acc_chip}
    </div>
    """)

    c1, c2, c3 = st.columns([1, 1, 1.25])
    with c1:
        if positive:
            html(f"""
            <div class="feature-card red">
                <div>
                    <div class="feature-top"><div class="feature-dot"></div><span class="feature-tag">{t('pill_result', lang)}</span></div>
                    <p class="feature-title" style="font-size:24px;">{t('pos_title', lang)}</p>
                    <p class="feature-text">{t('pos_subtitle', lang)}</p>
                </div>
                <div class="feature-bars"><span class="on"></span><span class="on"></span><span class="on"></span></div>
            </div>
            """)
        else:
            html(f"""
            <div class="feature-card">
                <div>
                    <div class="feature-top"><div class="feature-dot"></div><span class="feature-tag">{t('pill_result', lang)}</span></div>
                    <p class="feature-title" style="font-size:24px;">{t('neg_title', lang)}</p>
                    <p class="feature-text">{t('neg_subtitle', lang)}</p>
                </div>
                <div class="feature-bars"><span class="on"></span><span></span><span></span></div>
            </div>
            """)

    with c2:
        with st.container(key="cardr1_donut"):
            card_header(t('card_pred_prob', lang), t('pill_model_output', lang))
            st.plotly_chart(donut_chart(probability, positive, lang), width="stretch", config=PLOT_CONFIG)

    with c3:
        with st.container(key="cardr1_analysis"):
            card_header(t('analysis_title', lang),
                        t('chart_diab', lang) if positive else t('chart_non_diab', lang),
                        "pill-red" if positive else "pill-green")
            if positive:
                html(f"""
                <p style="font-size:14px; line-height:1.65;">{t('pos_analysis_desc', lang, prob=diabetes_pct)}</p>
                <div class="list-row"><div class="list-icon" style="background:#FDECEC;">👩‍⚕️</div><div>
                <p class="list-main">{t('pos_tip1_title', lang)}</p><p class="list-sub">{t('pos_tip1_sub', lang)}</p></div></div>
                <div class="list-row"><div class="list-icon" style="background:#FDECEC;">🥗</div><div>
                <p class="list-main">{t('pos_tip2_title', lang)}</p><p class="list-sub">{t('pos_tip2_sub', lang)}</p></div></div>
                """)
            else:
                html(f"""
                <p style="font-size:14px; line-height:1.65;">{t('neg_analysis_desc', lang)}</p>
                <div class="list-row"><div class="list-icon">🏃</div><div>
                <p class="list-main">{t('neg_tip1_title', lang)}</p><p class="list-sub">{t('neg_tip1_sub', lang)}</p></div></div>
                <div class="list-row"><div class="list-icon">📅</div><div>
                <p class="list-main">{t('neg_tip2_title', lang)}</p><p class="list-sub">{t('neg_tip2_sub', lang)}</p></div></div>
                """)

    render_result_explanation(probability, is_positive=positive)

    input_data = st.session_state.get('input_data', {})
    if input_data:
        with st.container(key="card_insights"):
            render_interactive_insights(input_data)

    # New prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(t('btn_new_pred', lang), key="new_pred_bottom", type="primary", width="stretch"):
            go_to("input")


def result_negative_page():
    result_page(positive=False)


def result_positive_page():
    result_page(positive=True)


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
