import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Risk Dashboard", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.big-title {
    font-size: 40px;
    font-weight: 700;
}
.subtext {
    color: gray;
    font-size: 16px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<p class="big-title">🏦 Loan Default Risk Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">AI-powered credit risk assessment system</p>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([1,1])

# ================= LEFT =================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Borrower Details")

    age = st.number_input("Age", 18, 100, 25)
    income = st.number_input("Annual Income", value=50000)
    loan_amount = st.number_input("Loan Amount", value=10000)
    emp_length = st.number_input("Employment Length", value=2)

    home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
    default_history = st.selectbox("Past Default?", ["N", "Y"])
    intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

    predict = st.button("🚀 Predict Risk")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RIGHT =================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict:

        with st.spinner("Analyzing borrower profile..."):
            import time
            time.sleep(1)

        # Derived feature
        loan_percent_income = loan_amount / income if income > 0 else 0

        # Create dataframe
        input_df = pd.DataFrame({
            'person_age': [age],
            'person_income': [income],
            'person_emp_length': [emp_length],
            'loan_amnt': [loan_amount],
            'loan_percent_income': [loan_percent_income],
            'person_home_ownership': [home],
            'loan_intent': [intent],
            'loan_grade': [grade],
            'cb_person_default_on_file': [default_history]
        })

        # Convert to dummies
        input_df = pd.get_dummies(input_df)

        # Align with training columns
        model_columns = model.feature_names_in_
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Prediction
        prob = model.predict_proba(input_df)[0][1]

        # Metric
        st.metric("Default Probability", f"{prob:.2%}")

        # Risk label
        if prob > 0.7:
            st.error("🔴 High Risk Borrower")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk Borrower")
        else:
            st.success("🟢 Low Risk Borrower")

        # -----------------------------
        # GAUGE CHART
        # -----------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Default Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # INSIGHT
        # -----------------------------
        st.markdown("### 💡 Insight")
        st.info("Higher loan burden and past defaults increase risk.")

    else:
        st.info("Enter borrower details and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# DASHBOARD SECTION
# -----------------------------
st.markdown("## 📊 Risk Analysis Dashboard")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Feature Comparison")
    chart_data = pd.DataFrame({
        "Feature": ["Income", "Loan Amount", "Employment"],
        "Value": [income, loan_amount, emp_length]
    })
    st.bar_chart(chart_data.set_index("Feature"))

with col4:
    st.subheader("Key Risk Drivers")
    st.write("• Loan burden (loan / income)")
    st.write("• Past default history")
    st.write("• Income level")
