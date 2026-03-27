import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
from src.db import log_prediction, load_predictions
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        padding-top: 2rem;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 8px 0;
    }

    /* Main background */
    .main { background-color: #f8fafc; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Page title */
    h1 { color: #0f172a !important; font-weight: 700 !important; }
    h2, h3 { color: #1e293b !important; font-weight: 600 !important; }

    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    /* Insight badges */
    .badge-red {
        background: #fef2f2; color: #991b1b;
        padding: 10px 16px; border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 8px 0; font-size: 14px;
    }
    .badge-green {
        background: #f0fdf4; color: #166534;
        padding: 10px 16px; border-radius: 8px;
        border-left: 4px solid #22c55e;
        margin: 8px 0; font-size: 14px;
    }
    .badge-amber {
        background: #fffbeb; color: #92400e;
        padding: 10px 16px; border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 8px 0; font-size: 14px;
    }

    /* Predict button */
    .stButton > button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #1d4ed8 !important;
    }

    /* Divider */
    hr { border-color: #e2e8f0; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load Data & Model ────────────────────────────────────────────────────────
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

df, rfm         = load_data()
model, features = load_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Churn Intelligence")
    st.markdown("---")
    page = st.radio("", [
        "Churn Predictor",
        "Overview",
        "EDA & Insights",
        "Customer Segments"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#94a3b8; line-height:1.8'>
    <b style='color:#cbd5e1'>Model Info</b><br>
    Algorithm: XGBoost<br>
    ROC-AUC: 0.9969<br>
    CV AUC: 0.9984<br>
    Dataset: 5,630 customers<br>
    Features: 18
    </div>
    """, unsafe_allow_html=True)
    # ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHURN PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
if page == "Churn Predictor":
    st.markdown("# Customer Churn Predictor")
    st.markdown("##### Enter customer details to get an instant churn risk assessment")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Account Details**")
        tenure       = st.slider("Tenure (months)", 0, 60, 12)
        city_tier    = st.selectbox("City Tier", [1,2,3],
                       format_func=lambda x: f"Tier {x} — {'Metro' if x==1 else 'Mid-size' if x==2 else 'Small'}")
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        complain     = st.selectbox("Filed a Complaint?",
                       [0,1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        st.markdown("**Order Behaviour**")
        order_count     = st.slider("Order Count", 1, 20, 5)
        days_last_order = st.slider("Days Since Last Order", 0, 30, 5)
        cashback        = st.slider("Cashback Amount ($)", 0, 300, 150)
        warehouse_dist  = st.slider("Warehouse Distance (km)", 5, 100, 30)

    with col3:
        st.markdown("**App Engagement**")
        hour_spend  = st.slider("Hours on App / day", 0, 5, 2)
        devices     = st.slider("Registered Devices", 1, 6, 2)
        num_address = st.slider("Saved Addresses", 1, 10, 3)
        order_hike  = st.slider("Order Amount Hike (%)", 10, 30, 15)

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("Generate Churn Risk Assessment")

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

        st.markdown("---")
        st.markdown("### Risk Assessment Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Risk Level",
                  "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low")
        c3.metric("Prediction",
                  "Will Churn" if pred == 1 else "Will Stay")
        c4.metric("Confidence", f"{max(prob, 1-prob):.1%}")
        try:
            risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
            log_prediction(
                tenure, city_tier, satisfaction,
                complain, order_count, cashback,
                round(float(prob), 4), risk,
                "Churn" if pred == 1 else "Stay"
            )
            st.success("Prediction logged to database!")
        except Exception as e:
            st.warning(f"DB logging skipped: {e}")

    # ✅ PREDICTION HISTORY SECTION (BOTTOM OF PAGE)
    st.markdown("---")
    st.markdown("### Prediction History")

    try:
        history = load_predictions()
        if len(history) > 0:
            st.dataframe(
                history[['tenure','city_tier','satisfaction',
                         'churn_prob','risk_level',
                         'prediction','created_at']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No predictions logged yet.")
    except Exception as e:
        st.info("Make predictions above to see history here.")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.3 else "#22c55e"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={'suffix': '%', 'font': {'size': 36, 'color': '#0f172a'}},
                title={'text': "Churn Risk Score",
                       'font': {'size': 16, 'color': '#64748b'}},
                gauge={
                    'axis': {'range': [0, 100],
                             'tickcolor': '#94a3b8'},
                    'bar':  {'color': color, 'thickness': 0.25},
                    'bgcolor': 'white',
                    'bordercolor': '#e2e8f0',
                    'steps': [
                        {'range': [0,  30], 'color': '#f0fdf4'},
                        {'range': [30, 60], 'color': '#fffbeb'},
                        {'range': [60,100], 'color': '#fef2f2'}
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 3},
                        'thickness': 0.8,
                        'value': prob * 100
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='white',
                height=280,
                margin=dict(t=40, b=10, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Recommended Actions")
            if prob > 0.6:
                st.markdown("""
                <div class='badge-red'>Immediate intervention required</div>
                <div class='badge-amber'>Offer a personalised discount or cashback reward</div>
                <div class='badge-amber'>Assign a dedicated support agent if complaint exists</div>
                <div class='badge-amber'>Send re-engagement campaign within 48 hours</div>
                """, unsafe_allow_html=True)
            elif prob > 0.3:
                st.markdown("""
                <div class='badge-amber'>Monitor this customer closely</div>
                <div class='badge-green'>Send a loyalty reward to increase satisfaction</div>
                <div class='badge-green'>Encourage app engagement with push notifications</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='badge-green'>Customer is healthy — low churn risk</div>
                <div class='badge-green'>Continue current engagement strategy</div>
                <div class='badge-green'>Consider upselling premium features</div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
elif page == "Overview":
    st.markdown("# E-Commerce Churn Intelligence")
    st.markdown("##### Predict · Segment · Retain — powered by XGBoost & RFM Analysis")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", "5,630")
    c2.metric("Churn Rate",      f"{df['Churn'].mean():.1%}")
    c3.metric("Model AUC",       "0.9969")
    c4.metric("5-Fold CV AUC",   "0.9984")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Key Insights")
        st.markdown("""
        <div class='badge-red'>Customers with low tenure (&lt;6 months) are 3x more likely to churn</div>
        <div class='badge-red'>Customers who complained have a 25% higher churn rate</div>
        <div class='badge-amber'>City Tier 1 (Metro) shows the highest churn concentration</div>
        <div class='badge-green'>Higher cashback amount significantly reduces churn probability</div>
        <div class='badge-green'>Customers with 4-5 satisfaction score rarely churn</div>
        """, unsafe_allow_html=True)

    with col2:
        fig = px.pie(
            df,
            names=df['Churn'].map({0: 'Retained', 1: 'Churned'}),
            title='Customer Retention Overview',
            color_discrete_sequence=['#2563eb', '#ef4444'],
            hole=0.55
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Inter', size=13),
            legend=dict(orientation='h', y=-0.1),
            title_font_size=15,
            margin=dict(t=40, b=10)
        )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Project Pipeline")
    cols = st.columns(6)
    steps = [
        ("01", "Data Collection", "5,630 customers · 19 features"),
        ("02", "EDA",             "Missing values · distributions"),
        ("03", "ML Model",        "XGBoost · class balancing"),
        ("04", "Explainability",  "SHAP feature importance"),
        ("05", "Segmentation",    "RFM · K-Means · 4 clusters"),
        ("06", "Dashboard",       "Streamlit · Plotly · live demo"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class='card' style='text-align:center; padding:16px'>
            <div style='font-size:11px;color:#2563eb;font-weight:700;
                        letter-spacing:0.1em'>STEP {num}</div>
            <div style='font-size:14px;font-weight:600;
                        color:#0f172a;margin:6px 0'>{title}</div>
            <div style='font-size:11px;color:#64748b'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "EDA & Insights":
    st.markdown("# Exploratory Data Analysis")
    st.markdown("##### Understanding what drives customer churn")
    st.markdown("---")

    COLORS = {'Retained': '#2563eb', 'Churned': '#ef4444'}

    col1, col2 = st.columns(2)
    with col1:
        data = df.groupby('CityTier')['Churn'].mean().reset_index()
        data['CityTier'] = data['CityTier'].map({1:'Tier 1 (Metro)',2:'Tier 2 (Mid)',3:'Tier 3 (Small)'})
        fig = px.bar(data, x='CityTier', y='Churn',
                     title='Churn Rate by City Tier',
                     color='Churn', color_continuous_scale='Blues',
                     labels={'Churn':'Churn Rate','CityTier':''})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df,
                     x=df['Churn'].map({0:'Retained',1:'Churned'}),
                     y='Tenure',
                     color=df['Churn'].map({0:'Retained',1:'Churned'}),
                     title='Tenure Distribution by Churn Status',
                     color_discrete_map=COLORS,
                     labels={'x':'','y':'Tenure (months)'})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        data = df.groupby('SatisfactionScore')['Churn'].mean().reset_index()
        fig = px.bar(data, x='SatisfactionScore', y='Churn',
                     title='Churn Rate by Satisfaction Score',
                     color='Churn', color_continuous_scale='RdYlGn_r',
                     labels={'Churn':'Churn Rate','SatisfactionScore':'Score (1=Low · 5=High)'})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data = df.groupby('Complain')['Churn'].mean().reset_index()
        data['Complain'] = data['Complain'].map({0:'No Complaint',1:'Complained'})
        fig = px.bar(data, x='Complain', y='Churn',
                     title='Churn Rate by Complaint Status',
                     color='Churn', color_continuous_scale='Reds',
                     labels={'Churn':'Churn Rate','Complain':''})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.image('data/roc_curve.png',
                 caption='ROC Curve — AUC 0.9969',
                 use_container_width=True)
    with c2:
        st.image('data/shap_importance.png',
                 caption='SHAP Feature Importance',
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEGMENTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Customer Segments":
    st.markdown("# RFM Customer Segmentation")
    st.markdown("##### K-Means clustering on Recency · Frequency · Monetary value")
    st.markdown("---")

    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    churn_seg  = rfm.groupby('Segment')['Churn'].mean().reset_index()
    churn_seg.columns = ['Segment', 'Churn Rate']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, names='Segment', values='Count',
                     title='Segment Distribution',
                     color_discrete_sequence=['#2563eb','#7c3aed','#0891b2','#059669'],
                     hole=0.5)
        fig.update_layout(paper_bgcolor='white',
                          font=dict(family='Inter', size=13),
                          legend=dict(orientation='h', y=-0.1),
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(churn_seg.sort_values('Churn Rate', ascending=False),
                     x='Segment', y='Churn Rate',
                     title='Churn Rate by Segment',
                     color='Churn Rate',
                     color_continuous_scale='Reds',
                     labels={'Churn Rate':'Churn Rate','Segment':''})
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          font=dict(family='Inter'),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Segment Summary Table")
    seg_summary = rfm.groupby('Segment').agg(
        Customers    =('CustomerID', 'count'),
        Avg_Recency  =('Recency',    'mean'),
        Avg_Frequency=('Frequency',  'mean'),
        Avg_Monetary =('Monetary',   'mean'),
        Churn_Rate   =('Churn',      'mean')
    ).round(2).reset_index()
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3D Cluster Visualization")
    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                        color='Segment', opacity=0.65,
                        color_discrete_sequence=['#2563eb','#7c3aed','#0891b2','#059669'],
                        title='RFM Clusters in 3D Space')
    fig.update_layout(paper_bgcolor='white',
                      font=dict(family='Inter'),
                      margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)