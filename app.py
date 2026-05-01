import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    /* Add a bit of space at the top */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style the predict button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
        color: white;
    }
    /* Style metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* Dark mode adjustments for metric cards */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background-color: #262730;
            border: 1px solid #333;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("churn-prediction", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ── Header Section ────────────────────────────────────
col1, col2 = st.columns([1, 6])
with col1:
    # Use a clean radar/chart icon
    st.markdown('<div style="text-align: center; font-size: 60px;">📊</div>', unsafe_allow_html=True)
with col2:
    st.title("Customer Churn Predictor Dashboard")
    st.write("Enter the customer's profile and service details to assess their risk of churning.")

st.markdown("---")

# ── Input Fields Layout ───────────────────────────────
# We use columns to group related fields
col_demo, col_service, col_billing = st.columns(3)

with col_demo:
    st.subheader("👤 Demographics")
    gender         = st.selectbox("Gender", ["Female", "Male"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner", ["No", "Yes"])
    dependents     = st.selectbox("Dependents", ["No", "Yes"])
    phone          = st.selectbox("Phone Service", ["No", "Yes"])
    num_services   = st.slider("Number of Services", 0, 9, 3)

with col_service:
    st.subheader("🌐 Service Details")
    tenure         = st.slider("Tenure (Months)", 0, 72, 12)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col_billing:
    st.subheader("💳 Billing & Payment")
    monthly        = st.number_input("Monthly Charges ($)", 0.0, 500.0, 65.0)
    total          = st.number_input("Total Charges ($)", 0.0, 15000.0, 800.0)
    cltv           = st.number_input("CLTV (Lifetime Value)", 2000.0, 10000.0, 5000.0)
    payment        = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless      = st.selectbox("Paperless Billing", ["No", "Yes"])

st.markdown("---")

# ── Feature Engineering & Encoding ────────────────────
avg_spend        = total / (tenure + 1)
is_mtm           = 1 if contract == "Month-to-month" else 0
is_fiber         = 1 if internet == "Fiber optic" else 0
is_echeck        = 1 if payment == "Electronic check" else 0

contract_enc  = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
internet_enc  = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet]
payment_enc   = {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
                 "Electronic check": 2, "Mailed check": 3}[payment]

senior_enc    = 1 if senior == "Yes" else 0
partner_enc   = 1 if partner == "Yes" else 0
depend_enc    = 1 if dependents == "Yes" else 0
phone_enc     = 1 if phone == "Yes" else 0
paper_enc     = 1 if paperless == "Yes" else 0
gender_enc    = 1 if gender == "Male" else 0

# ── Build Feature DataFrame ───────────────────────────
features = pd.DataFrame([{
    'Gender': gender_enc,
    'Senior Citizen': senior_enc,
    'Partner': partner_enc,
    'Dependents': depend_enc,
    'Tenure Months': tenure,
    'Phone Service': phone_enc,
    'Multiple Lines': 0,
    'Internet Service': internet_enc,
    'Online Security': 0,
    'Online Backup': 0,
    'Device Protection': 0,
    'Tech Support': 0,
    'Streaming TV': 0,
    'Streaming Movies': 0,
    'Contract': contract_enc,
    'Paperless Billing': paper_enc,
    'Payment Method': payment_enc,
    'Monthly Charges': monthly,
    'Total Charges': total,
    'CLTV': cltv,
    'Avg_monthSpend': avg_spend,
    'Num Service': num_services,
    'Is Month to Month': is_mtm,
    'Fiber': is_fiber
}])

# ── Prediction Section ────────────────────────────────
st.subheader("📈 Prediction Results")

predict_col, result_col = st.columns([1, 2])

with predict_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 Predict Churn Risk")
    
    # Show summary metrics before prediction
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Estimated Avg Monthly Spend", f"${avg_spend:.2f}")

if predict_clicked:
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    prob_pct    = probability * 100

    with result_col:
        # Create a beautiful Gauge Chart using Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability", 'font': {'size': 20}},
            number = {'suffix': "%", 'font': {'size': 36}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "rgba(0,0,0,0)"}, # Hide standard bar
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "lightgray",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.3)"}, # Green
                    {'range': [30, 70], 'color': "rgba(241, 196, 15, 0.3)"},  # Yellow
                    {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.3)"}   # Red
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'thickness': 0.75,
                    'value': prob_pct
                }
            }
        ))
        
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, width='stretch')

        # Call to action based on result
        if prediction == 1:
            st.error(f"⚠️ **High Churn Risk ({prob_pct:.1f}%)**: This customer is likely to leave.")
            st.info("💡 **Recommendation**: Consider a proactive retention offer or discount on a yearly contract.")
        else:
            st.success(f"✅ **Low Churn Risk ({prob_pct:.1f}%)**: This customer is likely to stay.")
            st.info("💡 **Recommendation**: Customer is stable. Consider upselling additional services.")
