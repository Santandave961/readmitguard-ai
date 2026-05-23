import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("data/patient_data.csv")

# ── Encode categoricals ────────────────────────────────────
cat_cols = ["gender", "discharge_to", "insurance", "a1c_result"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ── Features / Target ──────────────────────────────────────
X = df.drop("readmitted_30days", axis=1)
y = df["readmitted_30days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train XGBoost ──────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=" * 50)
print("PATIENT READMISSION PREDICTOR — Model Report")
print("=" * 50)
print(f"\nROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── SHAP explainer ─────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ── Save artifacts ─────────────────────────────────────────
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)
with open("models/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# Save feature names
feature_names = list(X.columns)
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n✅ Model + artifacts saved to /models/")