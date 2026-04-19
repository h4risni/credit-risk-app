import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Loan Risk Predictor", layout="centered")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center;'>🏦 Loan Default Risk Predictor</h1>
<p style='text-align: center; color: gray;'>AI-powered credit risk assessment tool</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown("## 📋 Enter Borrower Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, value=25)
    income = st.number_input("Annual Income", value=50000)
    employment = st.number_input("Employment Length (years)", 0, 40, value=2)

with col2:
    loan_amount = st.number_input("Loan Amount", value=10000)
    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    default_history = st.selectbox("Past Default?", ["Y", "N"])

col3, col4 = st.columns(2)

with col3:
    intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])

with col4:
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

# -----------------------------
# Derived Feature
# -----------------------------
loan_percent_income = loan_amount / income if income > 0 else 0

st.markdown("---")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Risk"):

    # Create input dataframe
    input_df = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [employment],
        'loan_amnt': [loan_amount],
        'loan_percent_income': [loan_percent_income],
        'person_home_ownership': [home],
        'loan_intent': [intent],
        'loan_grade': [grade],
        'cb_person_default_on_file': [default_history]
    })

    # Convert to dummies
    input_df = pd.get_dummies(input_df)

    # Align columns
    model_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict_proba(input_df)[0][1]

    # -----------------------------
    # OUTPUT SECTION (PREMIUM)
    # -----------------------------
    st.markdown("## 📊 Risk Assessment Result")

    # Metric
    st.metric("Default Probability", f"{round(prediction*100,2)}%")

    # Progress bar
    st.progress(float(prediction))

    # Risk classification
    if prediction > 0.5:
        st.error("⚠️ HIGH RISK")
    else:
        st.success("✅ LOW RISK")

    st.markdown("---")

    # -----------------------------
    # WHY SECTION (TOPPER MOVE)
    # -----------------------------
    st.markdown("### 🧠 Why this result?")

    st.write("""
    - Higher loan burden increases default probability  
    - Past default history negatively impacts creditworthiness  
    - Loan grade reflects underlying borrower risk  
    """)

    # -----------------------------
    # INSIGHT (SIGNATURE)
    # -----------------------------
    st.info("💡 Insight: Loan burden (loan amount relative to income) is the strongest driver of default risk.")

    st.markdown("---")
