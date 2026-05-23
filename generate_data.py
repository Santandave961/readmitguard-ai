import pandas as pd
import numpy as np

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

# Realistic readmission logic
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
    + 0.04 * (df["discharge_to"] == "AMA").astype(int)
    + 0.03 * (df["a1c_result"] == "Abnormal").astype(int)
)
prob = prob.clip(0, 1)
df["readmitted_30days"] = (np.random.rand(n) < prob).astype(int)

df.to_csv("/home/claude/patient_readmission/data/patient_data.csv", index=False)
print(f"Dataset saved: {len(df)} rows | Readmission rate: {df['readmitted_30days'].mean():.1%}")