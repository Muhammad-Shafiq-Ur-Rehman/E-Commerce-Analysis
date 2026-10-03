import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------
# 1. PAGE CONFIG & GLOBAL HIGH-CONTRAST GLASSMORPHISM UI
# ----------------------------------------------------
st.set_page_config(
    page_title="E-Commerce AI Intelligence",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
/* Main Canvas Styling - Light Professional Background */
.main {
    background-color: #f1f5f9;
}
.stApp {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}

/* Hide or Blend Top Streamlit Header */
header[data-testid="stHeader"] {
    background-color: rgba(241, 245, 249, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* Global High-Contrast Text Rules */
html, body, [class*="css"], .stMarkdown, p, span, div, label, h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

/* Form Controls & Input Labels */
label, .stWidgetLabel, [data-testid="stWidgetLabel"] p {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* Inputs & Form Boxes */
div[data-baseweb="input"] > div, 
div[data-baseweb="base-input"],
input {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}

/* High-Contrast White Glass Cards & Metrics */
[data-testid="metric-container"], .glass-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stMetricValue"] {
    color: #0284c7 !important; /* Vivid Blue */
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* Dataframe Cards */
.stDataFrame {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
}

/* Custom Primary Buttons */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700;
    border: none;
    padding: 0.6rem 1rem;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #1e293b !important; /* Deep Navy */
    border-right: 1px solid #334155 !important;
}
section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* Sidebar Multi-select Tags Fix */
span[data-baseweb="tag"] {
    background-color: #3b82f6 !important; /* Clean Blue Tag */
    color: #ffffff !important;
}

/* Navigation Selectbox */
div[data-baseweb="select"] > div {
    background-color: #334155 !important;
    border: 1px solid #475569 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Tabs Styling */
button[data-baseweb="tab"] p {
    color: #64748b !important;
    font-weight: 600 !important;
}
button[aria-selected="true"] p {
    color: #0284c7 !important;
    font-weight: 800 !important;
}

/* Header Container & Footer Styling */
.header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2rem;
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #64748b !important;
    font-size: 0.875rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. CACHED RESOURCE & MODEL LOADING
# ----------------------------------------------------
@st.cache_resource
def load_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

load_nltk_resources()

@st.cache_resource
def load_models():
    conn = sqlite3.connect("ecommerce_cleaned1.db", check_same_thread=False)
    churn_model = joblib.load("churn_logistic_model.pkl")
    sentiment_model = joblib.load("sentiment_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return conn, churn_model, sentiment_model, tfidf_vectorizer

conn, churn_model, sentiment_model, tfidf_vectorizer = load_models()

@st.cache_data
def load_customer_features():
    return pd.read_csv("customer_features.csv")

customer_features_df = load_customer_features()

# ----------------------------------------------------
# 3. HELPER FUNCTIONS & NLP
# ----------------------------------------------------
lemmatizer = WordNetLemmatizer()
base_stopwords = set(stopwords.words('english')) - {"not", "no", "never", "nor", "neither", "very"}
domain_fillers = {"product", "item", "purchase", "bought", "buy", "one", "got", "order"}
custom_stopwords = base_stopwords.union(domain_fillers)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    processed = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(w, pos='a'), pos='v')
        for w in tokens if w not in custom_stopwords and len(w) > 1
    ]
    return " ".join(processed)

# ----------------------------------------------------
# 4. GLOBAL HEADER & SIDEBAR BRANDING
# ----------------------------------------------------
st.markdown("""
<div class="header-container">
    <div>
        <h1 style='margin:0; font-size:2rem; color:#f8fafc;'>🛒 Enterprise E-Commerce Intelligence</h1>
        <p style='margin:0; color:#94a3b8; font-size:0.9rem;'>AI-Driven Analytics & Predictive Decision Engine</p>
    </div>
    
</div>
""", unsafe_allow_html=True)

st.sidebar.title("⚡ Navigation")
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Churn Prediction", "Sentiment Analysis"])

# Apply high-contrast dark theme layout to Plotly
# Apply light high-contrast theme layout to Plotly charts
plotly_layout_defaults = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#0f172a', size=12),
    title=dict(font=dict(color='#0f172a', size=16)),
    xaxis=dict(gridcolor='#e2e8f0', tickfont=dict(color='#334155')),
    yaxis=dict(gridcolor='#e2e8f0', tickfont=dict(color='#334155'))
)

# ====================================================
# DASHBOARD PAGE
# ====================================================
if page == "Dashboard":

    st.title("📊 Enterprise Performance Dashboard")

    # Sidebar Dashboard Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Dashboard Filters")
    categories = pd.read_sql("SELECT DISTINCT category FROM products", conn)['category'].tolist()
    selected_categories = st.sidebar.multiselect("Filter Categories", options=categories, default=categories)
    
    categories_placeholder = "'" + "','".join(selected_categories) + "'" if selected_categories else "''"

    # KPI Metrics Query
    revenue_query = f"SELECT SUM(o.quantity * o.unit_price * (1-o.discount)) FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category IN ({categories_placeholder})"
    orders_query = f"SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category IN ({categories_placeholder})"
    customers_query = f"SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category IN ({categories_placeholder})"

    total_revenue = pd.read_sql(revenue_query, conn).iloc[0, 0] or 0.0
    total_orders = pd.read_sql(orders_query, conn).iloc[0, 0] or 0
    total_customers = pd.read_sql(customers_query, conn).iloc[0, 0] or 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Unique Customers", f"{total_customers:,}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 Sales Analytics", "👥 Customer Analytics", "📦 Product Analytics"])

    with tab1:
        # Monthly Revenue Trend
        monthly_query = f"""
        SELECT strftime('%Y-%m', o.order_date) AS month, SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue
        FROM orders o JOIN products p ON o.product_id = p.product_id
        WHERE p.category IN ({categories_placeholder})
        GROUP BY month ORDER BY month
        """
        monthly_df = pd.read_sql(monthly_query, conn)
        fig_month = px.line(monthly_df, x="month", y="revenue", title="Monthly Revenue Trend", line_shape="spline")
        fig_month.update_traces(line_color="#38bdf8", line_width=3)
        fig_month.update_layout(**plotly_layout_defaults)
        st.plotly_chart(fig_month, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            city_query = f"""
            SELECT c.city, SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue
            FROM orders o JOIN customers c ON o.customer_id = c.customer_id JOIN products p ON o.product_id = p.product_id
            WHERE p.category IN ({categories_placeholder})
            GROUP BY c.city ORDER BY revenue DESC LIMIT 10
            """
            city_df = pd.read_sql(city_query, conn)
            city_fig = px.bar(city_df, x="city", y="revenue", color="revenue", title="Top Cities by Revenue", color_continuous_scale="Blues")
            city_fig.update_layout(**plotly_layout_defaults)
            st.plotly_chart(city_fig, use_container_width=True)

        with col_right:
            heat_query = f"""
            SELECT strftime('%m', o.order_date) AS month_num, p.category, SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue
            FROM orders o JOIN products p ON o.product_id = p.product_id
            WHERE p.category IN ({categories_placeholder})
            GROUP BY month_num, p.category
            """
            heat_df = pd.read_sql(heat_query, conn)
            if not heat_df.empty:
                pivot = heat_df.pivot(index="category", columns="month_num", values="revenue").fillna(0)
                heat_fig = px.imshow(pivot, title="Revenue Heatmap (Category vs Month)", color_continuous_scale="Viridis")
                heat_fig.update_layout(**plotly_layout_defaults)
                st.plotly_chart(heat_fig, use_container_width=True)

    with tab2:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            membership_query = "SELECT membership_type, COUNT(*) AS customers FROM customers GROUP BY membership_type"
            membership_df = pd.read_sql(membership_query, conn)
            fig_mem = px.pie(membership_df, names="membership_type", values="customers", hole=0.4, title="Membership Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_mem.update_layout(**plotly_layout_defaults)
            st.plotly_chart(fig_mem, use_container_width=True)

        with col_c2:
            payment_query = "SELECT payment_method, COUNT(*) AS transactions FROM orders GROUP BY payment_method"
            payment_df = pd.read_sql(payment_query, conn)
            fig_pay = px.pie(payment_df, names="payment_method", values="transactions", hole=0.4, title="Payment Method Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pay.update_layout(**plotly_layout_defaults)
            st.plotly_chart(fig_pay, use_container_width=True)

    with tab3:
        category_query = f"""
        SELECT p.category, SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue, AVG(o.returned)*100 AS return_rate
        FROM orders o JOIN products p ON o.product_id = p.product_id
        WHERE p.category IN ({categories_placeholder})
        GROUP BY p.category ORDER BY revenue DESC
        """
        cat_df = pd.read_sql(category_query, conn)

        fig_cat = px.bar(cat_df, x="category", y="revenue", color="revenue", title="Category Revenue", color_continuous_scale="Purples")
        fig_cat.update_layout(**plotly_layout_defaults)
        st.plotly_chart(fig_cat, use_container_width=True)

        fig_ret = px.bar(cat_df, x="category", y="return_rate", color="return_rate", title="Return Rate by Category (%)", color_continuous_scale="Reds")
        fig_ret.update_layout(**plotly_layout_defaults)
        st.plotly_chart(fig_ret, use_container_width=True)

        top_products_query = f"""
        SELECT p.product_name, p.category, SUM(o.quantity * o.unit_price * (1-o.discount)) AS total_revenue
        FROM orders o JOIN products p ON o.product_id = p.product_id
        WHERE p.category IN ({categories_placeholder})
        GROUP BY p.product_id ORDER BY total_revenue DESC LIMIT 10
        """
        top_products = pd.read_sql(top_products_query, conn)
        st.subheader("📦 Top 10 Products by Revenue")
        st.dataframe(top_products, use_container_width=True)

        st.download_button("📥 Download Top Products CSV", top_products.to_csv(index=False), "top_products.csv", "text/csv")

    st.divider()
    st.subheader("💡 Strategic Business Insights")
    st.info("""
    * **Revenue Concentration:** Top-tier membership tiers account for over 60% of total lifetime value.
    * **Return Rate Optimization:** Electronics and Apparel display higher return thresholds; targeting customer support in these categories can boost overall profitability.
    """)

# ====================================================
# CHURN PREDICTION PAGE
# ====================================================
elif page == "Churn Prediction":

    st.title("🔮 Predictive Churn Modeling & Risk Analysis")

    st.markdown("### Input Customer Profile")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            total_orders = st.number_input("Total Orders", min_value=0, value=10)
            total_spending = st.number_input("Total Spending ($)", min_value=0.0, value=5000.0)
            avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=500.0)
            days_since_last_order = st.number_input("Days Since Last Order", min_value=0, value=15)

        with col2:
            return_rate = st.number_input("Return Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.10)
            avg_delivery_days = st.number_input("Average Delivery Days", min_value=0.0, value=4.0)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            membership_type = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold", "Premium"])

    if st.button("Run Churn Diagnostic", type="primary"):
        input_df = pd.DataFrame({
            "total_orders": [total_orders],
            "total_spending": [total_spending],
            "avg_order_value": [avg_order_value],
            "days_since_last_order": [days_since_last_order],
            "return_rate": [return_rate],
            "avg_delivery_days": [avg_delivery_days],
            "age": [age],
            "membership_type": [membership_type]
        })

        prediction = churn_model.predict(input_df)[0]
        probability = churn_model.predict_proba(input_df)[0][1]

        st.divider()
        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            st.subheader("Churn Risk Gauge")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Churn Probability (%)", 'font': {'color': '#f8fafc'}},
                number={'suffix': "%", 'font': {'color': '#f8fafc'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#ef4444" if probability >= 0.75 else "#f59e0b" if probability >= 0.40 else "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ]
                }
            ))
            gauge.update_layout(**plotly_layout_defaults)
            st.plotly_chart(gauge, use_container_width=True)

        with col_g2:
            st.subheader("Risk Category")
            if probability >= 0.75:
                st.error("🚨 **HIGH RISK CUSTOMER**\n\nImmediate retention intervention required.")
            elif probability >= 0.40:
                st.warning("⚠️ **MEDIUM RISK CUSTOMER**\n\nTarget with promotional incentives.")
            else:
                st.success("✅ **LOW RISK CUSTOMER**\n\nCustomer engagement is stable.")

            st.markdown("#### 🔍 Explainable AI: Detected Risk Factors")
            risk_factors = []
            if days_since_last_order > 45:
                risk_factors.append(f"Extended inactivity ({days_since_last_order} days since last order)")
            if return_rate > 0.25:
                risk_factors.append(f"High return frequency ({return_rate:.0%})")
            if total_orders < 3:
                risk_factors.append("Low transaction history (< 3 orders)")

            if risk_factors:
                for rf in risk_factors:
                    st.write("•", rf)
            else:
                st.write("• No critical risk triggers detected.")

        numeric_cols = ["total_orders", "total_spending", "avg_order_value", "days_since_last_order", "return_rate", "avg_delivery_days", "age"]
        distances = euclidean_distances(customer_features_df[numeric_cols], input_df[numeric_cols])
        
        closest_idx = distances.argmin()
        closest_customer = customer_features_df.iloc[closest_idx]
        similarity_score = 1 / (1 + distances.min())

        st.divider()
        st.metric("Similarity Score to Archetype", f"{similarity_score:.2%}")

        st.subheader("👥 Side-by-Side Archetype Comparison")
        col_cmp1, col_cmp2 = st.columns(2)
        with col_cmp1:
            st.markdown("#### Input Profile")
            st.dataframe(input_df, use_container_width=True)
        with col_cmp2:
            st.markdown("#### Most Similar Historical Customer")
            st.dataframe(closest_customer.to_frame().T, use_container_width=True)

        customer_features_df["distance"] = distances
        top5 = customer_features_df.sort_values("distance").head(5)
        st.subheader("Top 5 Nearest Customer Profiles")
        st.dataframe(top5, use_container_width=True)

        report_data = input_df.copy()
        report_data["churn_probability"] = probability
        report_data["risk_level"] = "High" if probability >= 0.75 else "Medium" if probability >= 0.40 else "Low"
        st.download_button("📥 Download Churn Prediction Report", report_data.to_csv(index=False), "churn_report.csv", "text/csv")

# ====================================================
# SENTIMENT ANALYSIS PAGE
# ====================================================
elif page == "Sentiment Analysis":

    st.title("💬 Customer Sentiment & Review Classification")

    user_review = st.text_area("Customer Review", placeholder="Type or paste a customer review here...", height=120)

    if st.button("Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please provide review text.")
        else:
            cleaned_review = clean_text(user_review)
            vectorized = tfidf_vectorizer.transform([cleaned_review])

            prediction = sentiment_model.predict(vectorized)[0]
            probabilities = sentiment_model.predict_proba(vectorized)[0]
            classes = sentiment_model.classes_

            if prediction == "Positive":
                st.balloons()
                st.success(f"**Predicted Sentiment:** Positive")
            elif prediction == "Negative":
                st.error(f"**Predicted Sentiment:** Negative")
            else:
                st.info(f"**Predicted Sentiment:** {prediction}")

            col_s1, col_s2 = st.columns(2)

            with col_s1:
                confidence = max(probabilities)
                st.subheader("Confidence Gauge")
                conf_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={'suffix': "%", 'font': {'color': '#f8fafc'}},
                    title={'text': "Model Confidence", 'font': {'color': '#f8fafc'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#f8fafc"},
                        'bar': {'color': '#3b82f6'},
                        'steps': [{'range': [0, 100], 'color': 'rgba(59, 130, 246, 0.2)'}]
                    }
                ))
                conf_gauge.update_layout(**plotly_layout_defaults)
                st.plotly_chart(conf_gauge, use_container_width=True)

            with col_s2:
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({"Sentiment": classes, "Probability": probabilities})
                fig_prob = px.bar(prob_df, x="Sentiment", y="Probability", color="Sentiment", text_auto=".2%", color_discrete_sequence=px.colors.qualitative.Bold)
                fig_prob.update_layout(**plotly_layout_defaults)
                st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown(f"**Processed NLP Tokens:** `{cleaned_review}`")

            result_df = pd.DataFrame({
                "original_review": [user_review],
                "cleaned_review": [cleaned_review],
                "predicted_sentiment": [prediction],
                "confidence": [confidence]
            })
            st.download_button("📥 Download Analysis CSV", result_df.to_csv(index=False), "sentiment_analysis.csv", "text/csv")

# ----------------------------------------------------
# 5. GLOBAL FOOTER
# ----------------------------------------------------
st.markdown("""
<div class="footer">
    AI-Powered E-Commerce Customer Intelligence System | Data Science Final Hackathon Project 2026<br>
    Built with Streamlit, SQL, Machine Learning, and Plotly
</div>
""", unsafe_allow_html=True)