import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="SugarSense - Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Logo styling */
    .logo-text {
        font-size: 28px;
        font-weight: 600;
        color: #5B9AA9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .logo-icon {
        font-size: 32px;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 20px;
        opacity: 0.9;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .info-title {
        font-size: 28px;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    
    .info-text {
        font-size: 16px;
        color: #5B9AA9;
        line-height: 1.6;
        background: #E8F4F8;
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #A67C7C;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 15px;
    }
    
    /* Input form styling */
    .input-container {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 2rem auto;
    }
    
    .input-title {
        text-align: center;
        font-size: 32px;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 2rem;
    }
    
    /* Result styling */
    .result-positive {
        background: #A67C7C;
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .result-negative {
        background: #5B9AA9;
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .result-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-subtitle {
        font-size: 16px;
        opacity: 0.9;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .chart-title {
        font-size: 24px;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 1rem;
    }
    
    .chart-text {
        font-size: 15px;
        color: #666;
        line-height: 1.6;
    }
    
    /* Recommendation box */
    .recommendation {
        background: #5B9AA9;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: #5B9AA9;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #4A8999;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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

# Navigation
def navigation():
    col1, col2, col3 = st.columns([2, 6, 2])
    
    with col1:
        st.markdown("""
        <div class="logo-text">
            <span class="logo-icon">🩺</span>
            <span>SugarSense</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🏠 Home", key="nav_home"):
            st.session_state.page = "home"
            st.rerun()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Home/About Page
def home_page():
    navigation()
    
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">About<br>🩺 SugarSense</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # What is SugarSense
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2 class="info-title">What is<br>Sugar<span style="color: #5B9AA9;">Sense</span>?</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <p class="info-text">
                SugarSense is a simple AI-powered web application designed to help users 
                estimate their diabetes risk based on basic health indicators. By analyzing 
                information such as glucose level, BMI, and blood pressure, the system provides 
                an early prediction to encourage better health awareness and lifestyle habits.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # What SugarSense do
    col1, col2 = st.columns([1, 1])
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h2 class="info-title">What<br>Sugar<span style="color: #5B9AA9;">Sense</span><br>do?</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <p class="info-text">
                We use machine learning technology to analyze your health data and 
                generate a quick diabetes risk prediction in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Why SugarSense
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h2 class="info-title">Why<br>Sugar<span style="color: #5B9AA9;">Sense</span>?</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <p class="info-text">
                Early awareness can make a big difference. SugarSense aims to make health 
                screening more accessible, simple, and easy to understand for everyone.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h3 style="margin-bottom: 0.5rem;">⚠️ Disclaimer</h3>
        <p style="margin: 0;">
            SugarSense is intended for educational purposes only and should not replace professional 
            medical advice or diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #5B9AA9; margin-top: 2rem;'>Stay aware. Stay healthy 💙</p>", unsafe_allow_html=True)
    
    # Start button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Prediction", key="start_prediction", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# Input Page
def input_page():
    navigation()
    
    st.markdown("""
    <div class="input-container">
        <h1 class="input-title">Put Your Data Here</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
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
            submit = st.form_submit_button("🔬 Get Prediction!", use_container_width=True)
    
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

# Result Page - Negative
def result_negative_page():
    navigation()
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #2C3E50; font-size: 28px;">Your result is ...</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-negative">
        <h1 class="result-title">Great news,<br>No Diabetes Detected!</h1>
        <p class="result-subtitle">Your health indicators currently show a low likelihood of diabetes</p>
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
            hole=.6,
            marker=dict(colors=['#8B7FD9', '#F08080']),
            textinfo='none'
        )])
        
        fig.add_annotation(
            text=f"<b>{int(non_diabetes_prob)}</b>",
            font=dict(size=40, color='#2C3E50'),
            showarrow=False
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            height=350,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h3 class="chart-title">Looks like your sweetness level is just right</h3>
            <p class="chart-text">
                Based on your health data, the system detected a low possibility of diabetes. 
                The chart beside shows your current prediction result generated by our AI model. 
                Keep maintaining your healthy habits and continue taking care of your body. 💙
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation">
        💡 Keep up your healthy lifestyle and regular health check-ups
    </div>
    """, unsafe_allow_html=True)
    
    # New prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 New Prediction", key="new_pred_neg", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# Result Page - Positive
def result_positive_page():
    navigation()
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #2C3E50; font-size: 28px;">Your result is ...</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-positive">
        <h1 class="result-title">Oh no,<br>Diabetes Detected!</h1>
        <p class="result-subtitle">The model identified indicators associated with diabetes</p>
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
            hole=.6,
            marker=dict(colors=['#A67C7C', '#8B7FD9']),
            textinfo='none'
        )])
        
        fig.add_annotation(
            text=f"<b>{int(diabetes_prob)}</b>",
            font=dict(size=40, color='#2C3E50'),
            showarrow=False
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            height=350,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h3 class="chart-title">Looks like your sweetness level is a little too high</h3>
            <p class="chart-text">
                Based on your health data, you have a <b>{:.0f}% possibility</b> of diabetes. 
                The chart beside shows your current prediction result generated by our AI model.<br><br>
                Don't worry too much — this is only an early prediction to help you stay more aware of your health.
            </p>
        </div>
        """.format(probability * 100), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation">
        💡 We recommend you to consult a healthcare professional
    </div>
    """, unsafe_allow_html=True)
    
    # New prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 New Prediction", key="new_pred_pos", use_container_width=True):
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
