<picture><img src="https://github.com/user-attachments/assets/b5dadb42-1dd9-4461-8284-d82c336f34a2" alt="Heart Disease Prediction" width="100%" /></picture>

<h1 align="center">Heart Disease Prediction</h1>

<p align="center"><strong>تطبيق Streamlit بواجهة عربية يتوقع احتمال مرض القلب من 13 مدخلًا سريريًا - an Arabic-first Streamlit app that predicts heart-disease likelihood from 13 clinical inputs.</strong></p>

<p align="center">
<img src="https://img.shields.io/badge/Python-0b1220?style=for-the-badge&logo=python&logoColor=67e8f9" alt="Python" />
<img src="https://img.shields.io/badge/Streamlit-0b1220?style=for-the-badge&logo=streamlit&logoColor=67e8f9" alt="Streamlit" />
<img src="https://img.shields.io/badge/scikit--learn-0b1220?style=for-the-badge&logo=scikit-learn&logoColor=67e8f9" alt="scikit-learn" />
</p>

<p align="center"><sub>Artwork is illustrative.</sub></p>

> Educational demo, not a medical device. A prediction here is never a diagnosis - real concerns need a doctor.

## What it does

A single-page Streamlit app (`app.py`): the user enters the classic 13 heart-disease
features through Arabic-labelled sliders and dropdowns (age, sex, chest-pain type,
resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart
rate, exercise-induced angina, oldpeak, slope, fluoroscopy vessel count,
thalassemia). The inputs are one-hot encoded, aligned to the training columns,
scaled with the saved `scaler.pkl`, and scored by the saved `heart_model.pkl`.
The result is shown as a clear Arabic success/warning message.

## Files

| File | What it is |
|---|---|
| `app.py` | The whole Streamlit app (UI + inference) |
| `heart_model.pkl` | Trained classifier (joblib) |
| `scaler.pkl` | Fitted scaler, carries the training column order |
| `HeartDiseaseTrain-Test.csv` | The training/test data the artifacts were built from |
| `requirements.txt` | Python dependencies |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

## Measured performance

The saved model (`heart_model.pkl`, a RandomForestClassifier) was evaluated on a
20% stratified hold-out of the shipped dataset (205 of 1,025 rows,
random_state=42, same one-hot + scaling pipeline the app uses):

| Metric | Value |
|---|---|
| Accuracy | 99.5% |
| F1 (positive class) | 0.995 |
| Confusion matrix | 100 true neg / 0 false pos / 1 false neg / 104 true pos |

## Honest notes

- The repo ships the inference app and artifacts, not the training notebook.
  The numbers above are measured on the shipped artifacts; near-perfect scores
  are common on this classic dataset and do not mean the model generalizes to
  real patients. The shipped scaler was fit on the full dataset, so the
  hold-out numbers are slightly optimistic (minor leakage through feature
  scaling). Treat predictions as a demo of the UI + pipeline, not as evidence.
- The dataset follows the classic UCI Heart Disease schema (Cleveland-style
  13 features + target).
