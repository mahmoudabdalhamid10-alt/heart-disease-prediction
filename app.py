import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("توقع مرض القلب - Heart Disease Prediction")

age = st.slider("العمر", 20, 100, 50)
sex = st.selectbox("الجنس", ["Male", "Female"])
chest_pain = st.selectbox("نوع ألم الصدر", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
resting_bp = st.slider("ضغط الدم", 80, 200, 120)
cholesterol = st.slider("الكوليسترول", 100, 600, 200)
fasting_bs = st.selectbox("سكر الدم صايم", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"])
rest_ecg = st.selectbox("رسم القلب أثناء الراحة", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_hr = st.slider("أعلى معدل ضربات قلب", 60, 220, 150)
exercise_angina = st.selectbox("ألم صدر مع المجهود", ["Yes", "No"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
vessels = st.selectbox("عدد الأوعية الملونة", ["Zero", "One", "Two", "Three"])
thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect", "No"])

if st.button("توقع"):
    raw_input = pd.DataFrame([{
        'age': age, 'resting_blood_pressure': resting_bp, 'cholestoral': cholesterol,
        'Max_heart_rate': max_hr, 'oldpeak': oldpeak,
        'sex': sex, 'chest_pain_type': chest_pain, 'fasting_blood_sugar': fasting_bs,
        'rest_ecg': rest_ecg, 'exercise_induced_angina': exercise_angina,
        'slope': slope, 'vessels_colored_by_flourosopy': vessels, 'thalassemia': thalassemia
    }])

    raw_encoded = pd.get_dummies(raw_input)
    raw_encoded = raw_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

    input_scaled = scaler.transform(raw_encoded)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("النتيجة: احتمال وجود مرض قلب")
    else:
        st.success("النتيجة: مفيش احتمال مرض قلب")