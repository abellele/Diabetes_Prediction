# 🩺 SugarSense - Diabetes Prediction Web App

SugarSense is an AI-powered web application designed to help users estimate their diabetes risk based on basic health indicators.

## 📋 Features

- **Bilingual Support**: Seamless toggle between **English 🇬🇧** and **Bahasa Indonesia 🇮🇩**
- **Interactive Home Page**: Learn about SugarSense and its purpose
- **Data Input Form**: Easy-to-use form with validation and helper text
- **Real-time Prediction**: Instant diabetes risk assessment powered by the revised Ensemble Stacking model
- **Interactive Health Insights**: Deep-dive recommendations, visual gauges, and What-If simulator
- **Visual Results**: Beautiful donut charts showing prediction probabilities
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository or download the files

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Make sure you have the trained model file `model.pkl` in the same directory

### Running the Application

Run the main app with the new UI:
```bash
streamlit run app.py
```

Or run the simple version:
```bash
streamlit run diabetes_predict.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Input Parameters

The application requires the following health indicators:

1. **Glucose Level** (mmol/L) - Normal range: 4.0-7.0
2. **Hypertensive** - Yes/No
3. **Systolic Blood Pressure** (mmHg) - Normal: <120
4. **Weight** (kg)
5. **Body Mass Index (BMI)** - Normal range: 18.5-24.9
6. **Diastolic Blood Pressure** (mmHg) - Normal: <80
7. **Pulse Rate** (bpm) - Normal range: 60-100

## 🎨 UI/UX Design

The application features:

- **Clean, modern interface** with smooth transitions
- **Color-coded results**: Blue for negative, Red/Brown for positive diabetes detection
- **Interactive charts** using Plotly for data visualization
- **Responsive layout** that adapts to different screen sizes
- **Helpful tooltips** and validation messages

## ⚠️ Disclaimer

SugarSense is intended for **educational purposes only** and should not replace professional medical advice or diagnosis. Always consult with a healthcare professional for proper medical evaluation.

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Model**: Ensemble Stacking (Random Forest, Logistic Regression, KNN, XGBoost)
- **Data Processing**: scikit-learn, NumPy, Pandas
- **Visualization**: Plotly

## 📝 Model Information

The prediction model uses the newest revised Ensemble Stacking approach from `(1)_model_balancing_seperti_code_awal_awal.ipynb`:
- **Model**: Ensemble Stacking (`RandomForest`, `LogisticRegression`, `KNN`, `XGBoost` meta-classified by `LogisticRegression`)
- **Optimal threshold**: `0.72` (tuned on the validation set with recall constraint >= 0.60)
- **Data balancing**: RandomOverSampler (0.2) + BorderlineSMOTE (0.3)
- **Scaling**: StandardScaler fitted on training features (`scaler.pkl`)
- **Features (all 14 inputs, no feature selection)**: `age`, `gender`, `pulse_rate`, `systolic_bp`, `diastolic_bp`, `glucose`, `height`, `weight`, `bmi`, `family_diabetes`, `hypertensive`, `family_hypertension`, `cardiovascular_disease`, `stroke`. This corresponds to the notebook's "Full Feature" Ensemble Stacking experiment.
- **Retraining**: Run `python train_model.py` to retrain and update `model.pkl`, `scaler.pkl`, and `model_metadata.json`.

## 📄 License

This project is for educational purposes as part of a thesis/research project.

## 👨‍💻 Author

Developed for thesis research on diabetes prediction using machine learning.

---

**Stay aware. Stay healthy 💙**
