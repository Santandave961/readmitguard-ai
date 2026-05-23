import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.ensemble import GradientBoostingClassifier

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="ReadmitGuard AI",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background: #f7f9fc; }

.metric-card {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid #2563eb;
    margin-bottom: 16px;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
.metric-label { font-size: 0.85rem; color: #64748b; margin-top: 4px; }

.risk-high {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
}
.risk-medium {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    border: 2px solid #eab308;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
}
.risk-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; font-family: 'DM Serif Display', serif; }
.risk-prob { font-size: 3rem; font-weight: 800; }
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1e3a5f;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { background: #1d4ed8; transform: translateY(-1px); }

[data-testid="stSidebar"] {
    background: #1e3a5f;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading / training ───────────────────────────────
@st.cache_resource
def load_or_train_model():
    """Train a GradientBoosting model on synthetic data (no XGBoost dependency needed on cloud)."""
    np.random.seed(42)
    n = 2000

    df = pd.DataFrame({
        "age": np.random.randint(18, 90, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "num_diagnoses": np.random.randint(1, 10, n),
        "num_medications": np.random.randint(1, 20, n),
        "num_procedures": np.random.randint(0, 8, n),
        "num_lab_tests": np.random.randint(1, 30, n),
        "time_in_hospital": np.random.randint(1, 14, n),
        "num_outpatient_visits": np.random.randint(0, 10, n),
        "num_inpatient_visits": np.random.randint(0, 5, n),
        "num_emergency_visits": np.random.randint(0, 4, n),
        "diabetes": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "hypertension": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "heart_disease": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "discharge_to": np.random.choice(["Home", "Rehab", "SNF", "AMA"], n, p=[0.6, 0.2, 0.15, 0.05]),
        "insurance": np.random.choice(["NHIS", "Private", "HMO", "Self-Pay"], n),
        "a1c_result": np.random.choice(["Normal", "Abnormal", "Not Tested"], n, p=[0.4, 0.35, 0.25]),
    })

    cat_cols = ["gender", "discharge_to", "insurance", "a1c_result"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    prob = (
        0.05
        + 0.004 * df["age"]
        + 0.03 * df["num_diagnoses"]
        + 0.02 * df["num_inpatient_visits"]
        + 0.025 * df["num_emergency_visits"]
        + 0.015 * df["time_in_hospital"]
        + 0.04 * df["diabetes"]
        + 0.03 * df["hypertension"]
        + 0.05 * df["heart_disease"]
        + 0.04 * (df["discharge_to"] == le_dict["discharge_to"].transform(["AMA"])[0]).astype(int)
        + 0.03 * (df["a1c_result"] == le_dict["a1c_result"].transform(["Abnormal"])[0]).astype(int)
    ).clip(0, 1)
    df["readmitted_30days"] = (np.random.rand(n) < prob).astype(int)

    X = df.drop("readmitted_30days", axis=1)
    y = df["readmitted_30days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    feature_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, le_dict, list(X.columns), auc, cm, fpr, tpr, feature_importance, X_test, y_test


model, le_dict, feature_names, auc, cm, fpr, tpr, feat_imp, X_test, y_test = load_or_train_model()


# ── Sidebar nav ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## :hospital: ReadmitGuard AI")
    st.markdown("*30-Day Readmission Risk Predictor*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Predict Risk", "Model Performance", "Feature Insights", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** Gradient Boosting")
    st.markdown(f"**ROC-AUC:** {auc:.3f}")
    st.markdown("**Dataset:** 2,000 patients")
    st.markdown("**Target:** 30-day readmission")


# ══════════════════════════════════════════════════════════
# PAGE 1 — PREDICT RISK
# ══════════════════════════════════════════════════════════
if page == "Predict Risk":
    st.markdown("# ReadmitGuard AI")
    st.markdown("#### Predict a patient's 30-day hospital readmission risk using clinical features.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Patient Demographics</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 90, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        insurance = st.selectbox("Insurance Type", ["NHIS", "Private", "HMO", "Self-Pay"])

        st.markdown('<div class="section-header">Comorbidities</div>', unsafe_allow_html=True)
        diabetes = st.checkbox("Diabetes", value=False)
        hypertension = st.checkbox("Hypertension", value=False)
        heart_disease = st.checkbox("Heart Disease", value=False)
        a1c_result = st.selectbox("HbA1c Result", ["Normal", "Abnormal", "Not Tested"])

    with col2:
        st.markdown('<div class="section-header">Hospital Visit Details</div>', unsafe_allow_html=True)
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 4)
        num_diagnoses = st.slider("Number of Diagnoses", 1, 10, 3)
        num_medications = st.slider("Number of Medications", 1, 20, 8)
        num_procedures = st.slider("Number of Procedures", 0, 8, 2)
        num_lab_tests = st.slider("Number of Lab Tests", 1, 30, 10)
        discharge_to = st.selectbox("Discharged To", ["Home", "Rehab", "SNF", "AMA"])

        st.markdown('<div class="section-header">Prior Visits</div>', unsafe_allow_html=True)
        num_outpatient_visits = st.slider("Outpatient Visits (past year)", 0, 10, 1)
        num_inpatient_visits = st.slider("Inpatient Visits (past year)", 0, 5, 1)
        num_emergency_visits = st.slider("Emergency Visits (past year)", 0, 4, 0)

    st.markdown("---")
    predict_btn = st.button("Predict Readmission Risk")

    if predict_btn:
        # Build input
        input_dict = {
            "age": age,
            "gender": le_dict["gender"].transform([gender])[0],
            "num_diagnoses": num_diagnoses,
            "num_medications": num_medications,
            "num_procedures": num_procedures,
            "num_lab_tests": num_lab_tests,
            "time_in_hospital": time_in_hospital,
            "num_outpatient_visits": num_outpatient_visits,
            "num_inpatient_visits": num_inpatient_visits,
            "num_emergency_visits": num_emergency_visits,
            "diabetes": int(diabetes),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "discharge_to": le_dict["discharge_to"].transform([discharge_to])[0],
            "insurance": le_dict["insurance"].transform([insurance])[0],
            "a1c_result": le_dict["a1c_result"].transform([a1c_result])[0],
        }
        input_df = pd.DataFrame([input_dict])[feature_names]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if prob >= 0.65:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                emoji = "🔴"
                advice = "Immediate care coordination recommended. Schedule follow-up within 7 days."
            elif prob >= 0.40:
                risk_class = "risk-medium"
                risk_label = "MODERATE RISK"
                emoji = "🟡"
                advice = "Monitor closely. Schedule follow-up within 14 days."
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                emoji = "🟢"
                advice = "Standard discharge protocol. Follow-up within 30 days."

            st.markdown(f"""
            <div class="{risk_class}">
                <div class="risk-title">{emoji} {risk_label}</div>
                <div class="risk-prob">{prob:.0%}</div>
                <div style="font-size:0.9rem; margin-top:8px; color:#374151;">30-Day Readmission Probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"**Clinical Recommendation:** {advice}")

        # Feature contribution bar chart
        st.markdown('<div class="section-header">Top Risk Drivers for This Patient</div>', unsafe_allow_html=True)
        contrib = {
            "Age": age / 90,
            "Days in Hospital": time_in_hospital / 14,
            "# Diagnoses": num_diagnoses / 10,
            "# Emergency Visits": num_emergency_visits / 4,
            "# Inpatient Visits": num_inpatient_visits / 5,
            "Heart Disease": int(heart_disease),
            "Diabetes": int(diabetes),
            "Hypertension": int(hypertension),
        }
        contrib_df = pd.DataFrame(list(contrib.items()), columns=["Factor", "Score"]).sort_values("Score", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#ef4444" if s > 0.5 else "#3b82f6" for s in contrib_df["Score"]]
        ax.barh(contrib_df["Factor"], contrib_df["Score"], color=colors, edgecolor="none", height=0.6)
        ax.set_xlabel("Relative Contribution", fontsize=10)
        ax.set_title("Patient Risk Factor Breakdown", fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 1.1)
        for i, (v, f) in enumerate(zip(contrib_df["Score"], contrib_df["Factor"])):
            ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Evaluation metrics for the Gradient Boosting classifier on held-out test data.")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{auc:.3f}</div><div class="metric-label">ROC-AUC Score</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{precision:.1%}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{recall:.1%}</div><div class="metric-label">Recall (Sensitivity)</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title("Receiver Operating Characteristic", fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Readmitted", "Readmitted"], fontsize=9)
        ax.set_yticklabels(["Not Readmitted", "Readmitted"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════
# PAGE 3 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "Feature Insights":
    st.markdown("# Feature Insights")
    st.markdown("Which clinical features drive readmission risk the most?")
    st.markdown("---")

    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    top_feat = feat_imp.head(12)
    colors = ["#2563eb" if i < 3 else "#93c5fd" for i in range(len(top_feat))]
    ax.barh(top_feat.index[::-1], top_feat.values[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Feature Importance Score", fontsize=10)
    ax.set_title("Top Predictors of 30-Day Readmission", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(top_feat.values[::-1]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9, color="#374151")
    blue_patch = mpatches.Patch(color="#2563eb", label="Top 3 features")
    light_patch = mpatches.Patch(color="#93c5fd", label="Other features")
    ax.legend(handles=[blue_patch, light_patch], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Feature Distribution: Readmitted vs Not Readmitted</div>', unsafe_allow_html=True)
    df_plot = X_test.copy()
    df_plot["readmitted"] = y_test.values

    num_features = ["age", "num_diagnoses", "time_in_hospital", "num_emergency_visits"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, feat in zip(axes, num_features):
        for label, color, name in [(0, "#3b82f6", "Not Readmitted"), (1, "#ef4444", "Readmitted")]:
            data = df_plot[df_plot["readmitted"] == label][feat]
            ax.hist(data, bins=20, alpha=0.6, color=color, label=name, density=True)
        ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Density", fontsize=8)
    axes[0].legend(fontsize=8)
    plt.suptitle("Feature Distributions by Readmission Status", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("# About ReadmitGuard AI")
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### What is this?
        **ReadmitGuard AI** is a machine learning–powered clinical decision support tool that predicts the probability of a patient being readmitted to hospital within **30 days** of discharge.

        Early identification of high-risk patients enables care teams to:
        - Schedule timely follow-up appointments
        - Optimise discharge planning
        - Reduce avoidable readmissions and costs
        - Improve patient outcomes

        ---
        ### Model Details
        | Attribute | Value |
        |---|---|
        | Algorithm | Gradient Boosting Classifier |
        | Training data | 2,000 synthetic patient records |
        | Features | 16 clinical & demographic features |
        | Target | 30-day hospital readmission |
        | Evaluation metric | ROC-AUC |

        ---
        ### Tech Stack
        `Python` · `scikit-learn` · `Streamlit` · `Matplotlib` · `Pandas` · `NumPy`

        ---
        ### Disclaimer
        This tool is built for **portfolio demonstration purposes** using synthetic data.
        It is **not** validated for clinical use. Always consult qualified medical professionals for patient care decisions.
        """)
    with c2:
        st.markdown("""
        ### Built by
        **Wisdom**
        Data Science Portfolio · NYSC Corper

        GitHub: [Santandave961](https://github.com/Santandave961)

        X: [@Santandave961](https://twitter.com/Santandave961)

        ---
        ### Target Sector
        - Nigerian HMOs
        - Health insurance companies (AXA Mansard, Hygeia)
        - Hospital systems
        - Health tech startups
        """)