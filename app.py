import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ── Load Data & Model ───────────────────────────────────
@st.cache_data
def load_data():
    df  = pd.read_csv('data/cleaned_data.csv')
    rfm = pd.read_csv('data/rfm_segments.csv')
    return df, rfm

@st.cache_resource
def load_model():
    with open('data/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

df, rfm = load_data()
model, features = load_model()

# ── Sidebar ─────────────────────────────────────────────
page = st.sidebar.radio("Navigation", [
    "Churn Predictor",
    "Overview",
    "EDA & Insights",
    "Customer Segments"
])

# ════════════════════════════════════════════════════════
# PAGE 1 — CHURN PREDICTOR
# ════════════════════════════════════════════════════════
if page == "Churn Predictor":
    st.title("Customer Churn Predictor")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure", 0, 60, 12)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        satisfaction = st.slider("Satisfaction", 1, 5, 3)
        complain = st.selectbox("Complaint", [0, 1])

    with col2:
        order_count = st.slider("Order Count", 1, 20, 5)
        days_last_order = st.slider("Days Since Last Order", 0, 30, 5)
        cashback = st.slider("Cashback", 0, 300, 150)
        warehouse_dist = st.slider("Warehouse Distance", 5, 100, 30)

    with col3:
        hour_spend = st.slider("App Usage", 0, 5, 2)
        devices = st.slider("Devices", 1, 6, 2)
        num_address = st.slider("Addresses", 1, 10, 3)
        order_hike = st.slider("Order Hike %", 10, 30, 15)

    predict = st.button("Predict")

    if predict:
        input_dict = {
            'Tenure': tenure,
            'PreferredLoginDevice': 1,
            'CityTier': city_tier,
            'WarehouseToHome': warehouse_dist,
            'PreferredPaymentMode': 3,
            'Gender': 1,
            'HourSpendOnApp': hour_spend,
            'NumberOfDeviceRegistered': devices,
            'PreferedOrderCat': 2,
            'SatisfactionScore': satisfaction,
            'MaritalStatus': 1,
            'NumberOfAddress': num_address,
            'Complain': complain,
            'OrderAmountHikeFromlastYear': order_hike,
            'CouponUsed': 2,
            'OrderCount': order_count,
            'DaySinceLastOrder': days_last_order,
            'CashbackAmount': cashback
        }

        input_data = pd.DataFrame([input_dict])[features]

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.subheader("Result")
        st.write(f"Churn Probability: {prob:.2%}")
        st.write("Prediction:", "Churn" if pred == 1 else "Stay")

        # ✅ SESSION STORAGE (REPLACEMENT FOR MYSQL)
        if 'history' not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            'tenure': tenure,
            'city_tier': city_tier,
            'satisfaction': satisfaction,
            'churn_prob': round(float(prob), 4),
            'risk_level': "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low",
            'prediction': "Churn" if pred == 1 else "Stay"
        })

    # ✅ HISTORY SECTION
    st.markdown("---")
    st.subheader("Prediction History")

    if 'history' in st.session_state and len(st.session_state.history) > 0:
        st.dataframe(pd.DataFrame(st.session_state.history),
                     use_container_width=True)
    else:
        st.info("No predictions yet — make one above!")

# ════════════════════════════════════════════════════════
# PAGE 2 — OVERVIEW
# ════════════════════════════════════════════════════════
elif page == "Overview":
    st.title("Overview")

    st.metric("Total Customers", len(df))
    st.metric("Churn Rate", f"{df['Churn'].mean():.2%}")

    fig = px.pie(df, names='Churn')
    st.plotly_chart(fig)

# ════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ════════════════════════════════════════════════════════
elif page == "EDA & Insights":
    st.title("EDA")

    fig = px.box(df, x='Churn', y='Tenure')
    st.plotly_chart(fig)

# ════════════════════════════════════════════════════════
# PAGE 4 — SEGMENTS
# ════════════════════════════════════════════════════════
elif page == "Customer Segments":
    st.title("Segments")

    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                        color='Segment')
    st.plotly_chart(fig)