import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------- Load Data & Model ----------------
data = pd.read_csv("thyroid_dataset.csv")

# Encode 'sex' as 0/1
data['sex'] = data['sex'].map({'M': 0, 'F': 1})

X = data.drop("class", axis=1)
y = data["class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ---------------- Streamlit UI ----------------
st.title("Thyroid Disease Prediction")

# ---------------- Reference Table ----------------
st.subheader("Reference Ranges for Blood Test Parameters")
reference_data = {
    "Parameter": ["Age", "Sex", "TSH (µIU/mL)", "T3 (ng/dL)", "T4 (µg/dL)", "T4U", "FTI"],
    "Normal Range": ["1-80 years", "M/F", "0.4 - 4.0", "80 - 200", "4.5 - 12.0", "0.5 - 2.0", "4 - 12"]
}
ref_df = pd.DataFrame(reference_data)
st.table(ref_df)

st.write("Enter your blood test values:")

# Input fields with ranges
age = st.number_input("Age (years)", min_value=1, max_value=80, value=30)
sex = st.selectbox("Sex", options=["M", "F"])
TSH = st.number_input("TSH (µIU/mL)", min_value=0.4, max_value=20.0, value=2.0)
T3 = st.number_input("T3 (ng/dL)", min_value=50.0, max_value=300.0, value=100.0)
T4 = st.number_input("T4 (µg/dL)", min_value=2.0, max_value=20.0, value=8.0)
T4U = st.number_input("T4U", min_value=0.5, max_value=2.0, value=1.0)
FTI = st.number_input("FTI", min_value=2.0, max_value=20.0, value=8.0)

# Convert inputs to dataframe
input_data = pd.DataFrame([[age, 0 if sex=="M" else 1, TSH, T3, T4, T4U, FTI]],
                          columns=X.columns)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    st.subheader(f"Predicted Class: {prediction}")
    st.write("Prediction Probabilities:")
    st.write(pd.DataFrame([prediction_proba], columns=model.classes_))
