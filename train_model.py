"""
Model Training Script for SugarSense Diabetes Risk Prediction
Based on the newest notebook: (1)_model_balancing_seperti_code_awal_awal.ipynb

Features:
- Outlier handling (IQR clipping)
- 70/10/20 Stratified train/validation/test split
- Feature selection via Weighted Voting (Correlation, Random Forest, L1, RFE) -> Top 7 features
- StandardScaler normalization
- Class balancing via RandomOverSampler (0.2) + BorderlineSMOTE (0.3)
- Stacking Ensemble (Random Forest, Logistic Regression, KNN, XGBoost)
- Validation set threshold tuning with recall constraint >= 0.60
- Model and scaler persistence (model.pkl, scaler.pkl, model_metadata.json)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from xgboost import XGBClassifier

def train_and_save():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"dataset.csv not found at {csv_path}")

    print(f"[1/6] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, delimiter=',')

    # Cast boolean columns
    bool_cols = [
        'family_diabetes',
        'hypertensive',
        'family_hypertension',
        'cardiovascular_disease',
        'stroke'
    ]
    df[bool_cols] = df[bool_cols].astype(bool)

    # Outlier clipping on continuous numeric columns
    num_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if df[col].nunique() > 2
    ]
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    # Binary encodings
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['diabetic'] = df['diabetic'].map({'No': 0, 'Yes': 1})

    X_full = df.drop('diabetic', axis=1)
    y_full = df['diabetic']

    # 70% Train, 10% Validation, 20% Test
    print("[2/6] Splitting data (70% train, 10% validation, 20% test)...")
    X_train_full, X_temp_full, y_train, y_temp = train_test_split(
        X_full,
        y_full,
        test_size=0.3,
        stratify=y_full,
        random_state=42
    )

    X_val_full, X_test_full, y_val, y_test = train_test_split(
        X_temp_full,
        y_temp,
        test_size=2/3,
        stratify=y_temp,
        random_state=42
    )

    # Selected top 7 features from the notebook's weighted ranking
    best_features = ['glucose', 'hypertensive', 'systolic_bp', 'diastolic_bp', 'bmi', 'weight', 'pulse_rate']
    print(f"[3/6] Selected features: {best_features}")

    # Standard scaling
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train_full)
    X_val_scaled = scaler_full.transform(X_val_full)
    X_test_scaled = scaler_full.transform(X_test_full)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_full.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val_full.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_full.columns)

    # Class balancing
    print("[4/6] Resampling with RandomOverSampler + BorderlineSMOTE...")
    ros = RandomOverSampler(sampling_strategy=0.2, random_state=42)
    X_ros, y_ros = ros.fit_resample(X_train_scaled, y_train)

    bsmote = BorderlineSMOTE(sampling_strategy=0.3, random_state=42)
    X_train_resampled, y_train_resampled = bsmote.fit_resample(X_ros, y_ros)

    X_train_selected = X_train_resampled[best_features]
    X_val_selected = X_val_scaled[best_features]
    X_test_selected = X_test_scaled[best_features]

    # Save dedicated scaler for the selected features
    feature_indices = [X_train_full.columns.get_loc(c) for c in best_features]
    selected_mean = scaler_full.mean_[feature_indices].tolist()
    selected_scale = scaler_full.scale_[feature_indices].tolist()

    selected_scaler = StandardScaler()
    selected_scaler.mean_ = np.array(selected_mean)
    selected_scaler.scale_ = np.array(selected_scale)
    selected_scaler.var_ = np.square(selected_scale)
    selected_scaler.n_features_in_ = len(best_features)
    selected_scaler.feature_names_in_ = np.array(best_features, dtype=object)

    scaler_path = os.path.join(base_dir, "scaler.pkl")
    joblib.dump(selected_scaler, scaler_path)
    print(f"      Saved scaler to: {scaler_path}")

    # Build Ensemble Stacking model
    print("[5/6] Training Ensemble Stacking classifier...")
    rf_base = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=50,
        random_state=42
    )
    lr_base = LogisticRegression(
        C=0.01,
        max_iter=1000,
        class_weight='balanced'
    )
    knn_base = KNeighborsClassifier(
        n_neighbors=100
    )
    xgb_base = XGBClassifier(
        n_estimators=50,
        max_depth=2,
        reg_alpha=15,
        reg_lambda=15,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

    stacking_model = StackingClassifier(
        estimators=[
            ('rf', rf_base),
            ('lr', lr_base),
            ('knn', knn_base),
            ('xgb', xgb_base)
        ],
        final_estimator=LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=1000
        ),
        passthrough=True
    )

    stacking_model.fit(X_train_selected, y_train_resampled)

    # Threshold tuning on validation set
    print("[6/6] Tuning optimal threshold on validation set (recall >= 0.60)...")
    val_prob = stacking_model.predict_proba(X_val_selected)[:, 1]

    best_threshold = 0.50
    best_f1 = 0.0
    min_required_recall = 0.60

    for threshold in np.arange(0.10, 0.91, 0.01):
        pred_temp = (val_prob >= threshold).astype(int)
        rec_temp = recall_score(y_val, pred_temp, zero_division=0)
        f1_temp = f1_score(y_val, pred_temp, zero_division=0)

        if rec_temp >= min_required_recall and f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = float(round(threshold, 2))

    # Evaluate on test set
    test_prob = stacking_model.predict_proba(X_test_selected)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, test_prob)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION SUMMARY:")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Validation F1    : {best_f1:.4f}")
    print(f"Test Accuracy    : {test_acc:.4f}")
    print(f"Test Precision   : {test_prec:.4f}")
    print(f"Test Recall      : {test_rec:.4f}")
    print(f"Test F1          : {test_f1:.4f}")
    print(f"Test ROC-AUC     : {test_auc:.4f}")
    print("=" * 50)

    # Save model.pkl
    model_path = os.path.join(base_dir, "model.pkl")
    joblib.dump(stacking_model, model_path)
    print(f"[OK] Model saved to: {model_path}")

    # Save metadata
    meta = {
        'model_name': 'Ensemble Stacking (Random Forest, Logistic Regression, KNN, XGBoost)',
        'source_notebook': '(1)_model_balancing_seperti_code_awal_awal.ipynb',
        'selected_features': best_features,
        'optimal_threshold': best_threshold,
        'metrics': {
            'validation_f1': round(best_f1, 4),
            'test_accuracy': round(test_acc, 4),
            'test_precision': round(test_prec, 4),
            'test_recall': round(test_rec, 4),
            'test_f1': round(test_f1, 4),
            'test_roc_auc': round(test_auc, 4)
        },
        'scaler_mean': selected_mean,
        'scaler_scale': selected_scale
    }

    meta_path = os.path.join(base_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata saved to: {meta_path}")

if __name__ == '__main__':
    train_and_save()
