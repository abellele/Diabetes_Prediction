import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

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

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except:
        return None

model = load_model()

# Navigation - Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align: left; padding: 1rem 0 2rem 1rem;'>
            <h1 style='color: #2B3674; font-size: 24px; font-weight: 800; display: flex; align-items: center; gap: 10px;'>
                <span style="background: #2E66F6; color: white; border-radius: 8px; padding: 5px 10px; font-size: 20px;">S</span>
                SugarSense
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        selection = st.radio(
            "Navigation",
            ["Dashboard", "Prediction"],
            index=0 if st.session_state.page == "home" else 1,
            label_visibility="collapsed"
        )
        
        if selection == "Dashboard" and st.session_state.page != "home":
            st.session_state.page = "home"
            st.rerun()
        elif selection == "Prediction" and st.session_state.page not in ["input", "result_negative", "result_positive"]:
            st.session_state.page = "input"
            st.rerun()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Home/About Page
def home_page():
    render_sidebar()
    
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title"><span style="color: #FFD700;">Good Day!</span></h1>
        <p class="hero-subtitle">Welcome to your early diabetes risk prediction dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # What is SugarSense
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2 class="info-title"><span style="color: #2E66F6;">What is SugarSense?</span></h2>
            <p class="info-text">
                SugarSense is a simple AI-powered web application designed to help users 
                estimate their diabetes risk based on basic health indicators. By analyzing 
                information such as glucose level, BMI, and blood pressure, the system provides 
                an early prediction to encourage better health awareness and lifestyle habits.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h2 class="info-title"><span style="color: #2E66F6;">What SugarSense do?</span></h2>
            <p class="info-text">
                We use machine learning technology to analyze your health data and 
                generate a quick diabetes risk prediction in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Why SugarSense & Disclaimer
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2 class="info-title"><span style="color: #2E66F6;">Why SugarSense?</span></h2>
            <p class="info-text">
                Early awareness can make a big difference. SugarSense aims to make health 
                screening more accessible, simple, and easy to understand for everyone.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="disclaimer">
            <h3><span style="color: #DD6B20;">⚠️ Educational Purposes Only</span></h3>
            <p style="color: #718096; margin-top: 10px;">
                SugarSense is intended for educational purposes only and should not replace professional 
                medical advice or diagnosis. Always consult a healthcare professional.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Start button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("+ Start New Prediction", key="start_prediction", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# Input Page
def input_page():
    render_sidebar()
    
    st.markdown("""
    <div class="input-container" style="max-width: 800px; margin: 2rem auto;">
        <h1 class="input-title"><span style="color: #2E66F6;">Health Information</span></h1>
        <p style="color: #718096; margin-bottom: 2rem;">Please fill in your current health metrics to get an accurate prediction.</p>
    """, unsafe_allow_html=True)
    
    # Create form
    with st.form("prediction_form", border=False):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            glucose = st.number_input(
                "Glucose Level (mmol/L)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                help="Normal range: 4.0-7.0 mmol/L"
            )
            
            hypertensive = st.selectbox(
                "Hypertensive",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
            
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=0.0,
                value=120.0,
                step=1.0,
                help="Normal: <120 mmHg"
            )
            
            weight = st.number_input(
                "Weight (kg)",
                min_value=0.0,
                value=60.0,
                step=0.1
            )
            
        with col2:
            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=0.0,
                value=22.0,
                step=0.1,
                help="Normal range: 18.5-24.9"
            )
            
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=0.0,
                value=80.0,
                step=1.0,
                help="Normal: <80 mmHg"
            )
            
            pulse_rate = st.number_input(
                "Pulse Rate (bpm)",
                min_value=0.0,
                value=70.0,
                step=1.0,
                help="Normal range: 60-100 bpm"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
        with col_btn2:
            submit = st.form_submit_button("Generate Prediction", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        # Validate inputs
        if glucose == 0 or pulse_rate == 0 or weight == 0 or systolic_bp == 0 or bmi == 0 or diastolic_bp == 0:
            st.error("⚠️ Please enter valid values for all fields. Values cannot be 0.")
        elif model is None:
            st.error("❌ Model not found. Please ensure model.pkl exists in the directory.")
        else:
            # Store input data in session state
            st.session_state.input_data = {
                'glucose': glucose,
                'hypertensive': hypertensive,
                'pulse_rate': pulse_rate,
                'weight': weight,
                'systolic_bp': systolic_bp,
                'bmi': bmi,
                'diastolic_bp': diastolic_bp
            }
            
            # Prepare input for prediction
            input_array = np.array([[
                glucose, hypertensive, pulse_rate, weight, systolic_bp, bmi, diastolic_bp
            ]])
            
            # Apply scaling
            training_mean = np.array([7.26, 0.134, 74.5, 55.2, 138.8, 23.8, 81.3])
            training_std = np.array([4.8, 0.34, 12.5, 11.8, 21.5, 5.2, 10.8])
            input_scaled = (input_array - training_mean) / training_std
            
            # Predict
            try:
                probability = model.predict_proba(input_scaled)[0][1]
                optimal_threshold = 0.35
                prediction = 1 if probability >= optimal_threshold else 0
                
                # Store results
                st.session_state.probability = probability
                st.session_state.prediction = prediction
                
                # Navigate to result page
                if prediction == 1:
                    st.session_state.page = "result_positive"
                else:
                    st.session_state.page = "result_negative"
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

def generate_health_insights(input_data):
    insights = []
    
    bmi = input_data.get('bmi', 0)
    if bmi < 18.5:
        insights.append("⚖️ **Weight Management:** Your BMI indicates you are underweight. Consider a balanced diet plan to reach a healthy weight.")
    elif 25.0 <= bmi < 30.0:
        insights.append("⚖️ **Weight Management:** Your BMI indicates you are overweight. A balanced diet and regular physical activity can help.")
    elif bmi >= 30.0:
        insights.append("⚖️ **Weight Management:** Your BMI falls into the obesity category. It is highly recommended to consult a healthcare provider for a plan.")
        
    glucose = input_data.get('glucose', 0)
    if glucose > 7.0:
        insights.append("🩸 **Blood Sugar:** Your glucose levels are above the normal range. Reducing intake of refined sugars may help manage your levels.")
    elif glucose < 4.0:
        insights.append("🩸 **Blood Sugar:** Your glucose levels are below normal. Ensure you are eating regular, balanced meals.")
        
    systolic = input_data.get('systolic_bp', 0)
    diastolic = input_data.get('diastolic_bp', 0)
    hypertensive = input_data.get('hypertensive', 0)
    if systolic >= 130 or diastolic >= 80 or hypertensive == 1:
        insights.append("🫀 **Blood Pressure:** Your blood pressure indicators are elevated. Reducing sodium intake, regular exercise, and stress management can help.")
        
    pulse = input_data.get('pulse_rate', 0)
    if pulse > 100:
        insights.append("❤️ **Heart Rate:** Your resting pulse rate is high. Cardiovascular exercise and stress management might be beneficial.")
    elif pulse < 60 and pulse > 0:
        insights.append("❤️ **Heart Rate:** Your resting pulse rate is lower than average. This can be normal for athletes, but monitor for dizziness.")
        
    if not insights:
        insights.append("✨ **General Health:** Your metrics generally look to be within typical ranges. Keep up the good work!")
        
    return insights

# Result Page - Negative
def result_negative_page():
    render_sidebar()
    
    st.markdown("""
    <div style="text-align: left; margin: 1rem 0;">
        <h2 style="color: #2B3674; font-size: 28px;">Prediction Report</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-negative">
        <h1 class="result-title" style="color: #05CD99;">No Diabetes Detected</h1>
        <p class="result-subtitle">Your health indicators currently show a low likelihood of diabetes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart and explanation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create donut chart
        probability = st.session_state.get('probability', 0)
        diabetes_prob = probability * 100
        non_diabetes_prob = 100 - diabetes_prob
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=['Non-Diabetes', 'Diabetes'],
            values=[non_diabetes_prob, diabetes_prob],
            hole=.7,
            marker=dict(colors=['#2E66F6', '#E2E8F0']),
            textinfo='none'
        )])
        
        fig.add_annotation(
            text=f"<b>{int(non_diabetes_prob)}%</b><br><span style='font-size:12px;color:#718096'>Safe</span>",
            font=dict(size=36, color='#2B3674'),
            showarrow=False
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=300,
            margin=dict(t=50, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container" style="height: 100%;">
            <h3 class="chart-title">Health Status Analysis</h3>
            <p class="chart-text">
                Based on your health data, the system detected a low possibility of diabetes. 
                The chart on the left illustrates your current prediction confidence generated by our AI model. 
                Keep maintaining your healthy habits and continue taking care of your body. 
            </p>
            <div class="highlight-box">
                💡 Keep up your healthy lifestyle and regular health check-ups.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    input_data = st.session_state.get('input_data', {})
    if input_data:
        st.markdown("<div class='info-card'><h3 style='color: #2B3674; margin-bottom: 1rem;'>📌 Personalized Health Insights</h3>", unsafe_allow_html=True)
        for insight in generate_health_insights(input_data):
            st.info(insight)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # New prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("New Prediction", key="new_pred_neg", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# Result Page - Positive
def result_positive_page():
    render_sidebar()
    
    st.markdown("""
    <div style="text-align: left; margin: 1rem 0;">
        <h2 style="color: #2B3674; font-size: 28px;">Prediction Report</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-positive" style="border: 1px solid #FED7D7;">
        <h1 class="result-title" style="color: #E53E3E;">Diabetes Detected</h1>
        <p class="result-subtitle">The model identified indicators associated with diabetes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart and explanation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create donut chart
        probability = st.session_state.get('probability', 0)
        diabetes_prob = probability * 100
        non_diabetes_prob = 100 - diabetes_prob
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=['Diabetes', 'Non-Diabetes'],
            values=[diabetes_prob, non_diabetes_prob],
            hole=.7,
            marker=dict(colors=['#E53E3E', '#E2E8F0']),
            textinfo='none'
        )])
        
        fig.add_annotation(
            text=f"<b>{int(diabetes_prob)}%</b><br><span style='font-size:12px;color:#718096'>Risk</span>",
            font=dict(size=36, color='#2B3674'),
            showarrow=False
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=300,
            margin=dict(t=50, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container" style="height: 100%;">
            <h3 class="chart-title">Health Status Analysis</h3>
            <p class="chart-text">
                Based on your health data, you have a <b>{:.0f}% possibility</b> of diabetes. 
                The chart on the left illustrates your current prediction confidence generated by our AI model.<br><br>
                Don't panic — this is an early algorithmic prediction to help you stay more aware of your health.
            </p>
            <div class="highlight-box" style="background: #FFF1F0; color: #C53030; border: 1px solid #FED7D7;">
                💡 We highly recommend consulting a healthcare professional.
            </div>
        </div>
        """.format(probability * 100), unsafe_allow_html=True)
    
    input_data = st.session_state.get('input_data', {})
    if input_data:
        st.markdown("<div class='info-card'><h3 style='color: #2B3674; margin-bottom: 1rem;'>📌 Personalized Health Insights</h3>", unsafe_allow_html=True)
        for insight in generate_health_insights(input_data):
            st.info(insight)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # New prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("New Prediction", key="new_pred_pos", use_container_width=True):
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
