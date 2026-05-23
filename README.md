# ReadmitGuard AI 🏥

**30-Day Hospital Readmission Risk Predictor**

A machine learning–powered Streamlit app that predicts the probability of a patient being readmitted to hospital within 30 days of discharge, using clinical and demographic features.

---

## Features

- **Risk Prediction** — Enter patient details and get an instant readmission risk score with clinical recommendations
- **Model Performance** — ROC-AUC curve, confusion matrix, accuracy, precision & recall
- **Feature Insights** — Global feature importance and distribution plots by readmission status
- **Clean UI** — Professional healthcare-themed interface with risk-level colour coding

## Tech Stack

| Layer | Tools |
|---|---|
| ML Model | Gradient Boosting (scikit-learn) |
| Explainability | Feature importance plots |
| Frontend | Streamlit |
| Data | Synthetic (2,000 patients, 16 features) |
| Visualization | Matplotlib |

## Local Setup

```bash
git clone https://github.com/Santandave961/readmitguard-ai
cd readmitguard-ai
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

1. Push to GitHub
2. Go to share.streamlit.io
3. Point to `app.py`
4. Add `runtime.txt` → `3.11`

## Target Use Cases

- Nigerian HMOs and health insurance companies (AXA Mansard, Hygeia, Reliance HMO)
- Hospital systems seeking to reduce avoidable readmissions
- Health tech startups building care coordination tools

---

> ⚠️ Built for portfolio demonstration using synthetic data. Not validated for clinical use.

**Built by Wisdom | [@Santandave961](https://twitter.com/Santandave961)**
