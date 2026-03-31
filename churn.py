import streamlit as st
import pickle
import numpy as np

# ── Load Model ────────────────────────────────────────
with open("churn-prediction.pkl", "rb") as f:
    model = pickle.load(f)

# ── Page Config ───────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📡")
st.title("📡 Customer Churn Predictor")
st.write("Fill in the customer details below to predict churn risk.")

# ── Input Fields ──────────────────────────────────────
st.subheader("Customer Information")

tenure         = st.slider("Tenure (Months)", 0, 72, 12)
monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
total          = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment        = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.selectbox("Partner", ["No", "Yes"])
dependents     = st.selectbox("Dependents", ["No", "Yes"])
phone          = st.selectbox("Phone Service", ["No", "Yes"])
paperless      = st.selectbox("Paperless Billing", ["No", "Yes"])
num_services   = st.slider("Number of Services", 0, 9, 3)

# ── Feature Engineering (same as training) ────────────
avg_spend        = total / (tenure + 1)
is_mtm           = 1 if contract == "Month-to-month" else 0
is_fiber         = 1 if internet == "Fiber optic" else 0
is_echeck        = 1 if payment == "Electronic check" else 0

# ── Encode Inputs ─────────────────────────────────────
contract_enc  = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
internet_enc  = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet]
payment_enc   = {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
                 "Electronic check": 2, "Mailed check": 3}[payment]
senior_enc    = 1 if senior == "Yes" else 0
partner_enc   = 1 if partner == "Yes" else 0
depend_enc    = 1 if dependents == "Yes" else 0
phone_enc     = 1 if phone == "Yes" else 0
paper_enc     = 1 if paperless == "Yes" else 0

# ── Build Feature Array ───────────────────────────────
features = np.array([[
    0,            # Gender (not in UI, defaulting to 0)
    senior_enc,   # Senior Citizen
    partner_enc,  # Partner
    depend_enc,   # Dependents
    tenure,       # Tenure Months
    phone_enc,    # Phone Service
    0,            # Multiple Lines
    internet_enc, # Internet Service
    0, 0, 0, 0, 0, 0, # Security, Backup, Device, Tech, TV, Movies
    contract_enc, # Contract
    paper_enc,    # Paperless Billing
    payment_enc,  # Payment Method
    monthly,      # Monthly Charges
    total,        # Total Charges
    0,            # CLTV
    avg_spend,    # Avg_monthSpend
    num_services, # Num Service
    is_mtm,       # Is Month to Month
    is_fiber      # Fiber
]])


# ── Predict ───────────────────────────────────────────
if st.button("🔮 Predict Churn"):
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    prob_pct    = round(probability * 100, 1)

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {prob_pct}% probability")
        st.write("This customer is likely to leave. Consider a retention offer.")
    else:
        st.success(f"✅ Low Churn Risk — {prob_pct}% probability")
        st.write("This customer is likely to stay.")

    # Probability bar
    st.progress(probability)
    st.caption(f"Churn probability: {prob_pct}%")
