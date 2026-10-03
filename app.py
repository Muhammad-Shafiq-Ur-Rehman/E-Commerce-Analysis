import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import re
import nltk

import numpy as np
import plotly.graph_objects as go

from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------
# 1. PAGE CONFIG (Must be called ONCE at top)
# ----------------------------------------------------
st.set_page_config(
    page_title="AI-Powered E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------------------------------
# 2. CACHED RESOURCE LOADING
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
# 3. HELPER FUNCTIONS
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
    processed = []
    for word in tokens:
        if word not in custom_stopwords and len(word) > 1:
            lemma = lemmatizer.lemmatize(word, pos='a')
            lemma = lemmatizer.lemmatize(lemma, pos='v')
            processed.append(lemma)
            
    return " ".join(processed)

# ----------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ----------------------------------------------------
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Churn Prediction", "Sentiment Analysis"]
)

# ====================================================
# DASHBOARD PAGE
# ====================================================
if page == "Dashboard":

    st.title("📊 E-Commerce Dashboard")

    revenue_query = "SELECT SUM(quantity * unit_price * (1-discount)) AS total_revenue FROM orders"
    orders_query = "SELECT COUNT(*) AS total_orders FROM orders"
    customers_query = "SELECT COUNT(*) AS total_customers FROM customers"

    total_revenue = pd.read_sql(revenue_query, conn).iloc[0, 0]
    total_orders = pd.read_sql(orders_query, conn).iloc[0, 0]
    total_customers = pd.read_sql(customers_query, conn).iloc[0, 0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Total Customers", f"{total_customers:,}")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📈 Sales Analytics",
        "👥 Customer Analytics",
        "📦 Product Analytics"
    ])

    with tab1:
        # Monthly Revenue Trend
        monthly_query = """
        SELECT
            strftime('%Y-%m', order_date) AS month,
            SUM(quantity * unit_price * (1-discount)) AS revenue
        FROM orders
        GROUP BY month
        ORDER BY month
        """
        monthly_df = pd.read_sql(monthly_query, conn)
        fig_month = px.line(monthly_df, x="month", y="revenue", title="Monthly Revenue Trend")
        st.plotly_chart(fig_month, use_container_width=True)

        # City Revenue
        city_query = """
        SELECT
            c.city,
            SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.city
        ORDER BY revenue DESC
        """
        city_df = pd.read_sql(city_query, conn)
        city_fig = px.bar(
            city_df,
            x="city",
            y="revenue",
            color="revenue",
            title="Revenue By City"
        )
        st.plotly_chart(city_fig, use_container_width=True)

    with tab2:
        membership_query = """
        SELECT
            membership_type,
            COUNT(*) AS customers
        FROM customers
        GROUP BY membership_type
        """
        membership_df = pd.read_sql(membership_query, conn)
        fig_mem = px.pie(
            membership_df,
            names="membership_type",
            values="customers",
            hole=0.4,
            title="Membership Distribution"
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    with tab3:
        # Return Rate
        return_query = """
        SELECT
            p.category,
            AVG(o.returned)*100 AS return_rate
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.category
        """
        return_df = pd.read_sql(return_query, conn)
        return_fig = px.bar(
            return_df,
            x="category",
            y="return_rate",
            color="return_rate",
            title="Return Rate By Category (%)"
        )
        st.plotly_chart(return_fig, use_container_width=True)

        # Category Revenue
        category_query = """
        SELECT
            p.category,
            SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.category
        ORDER BY revenue DESC
        """
        category_df = pd.read_sql(category_query, conn)
        fig_cat = px.bar(category_df, x="category", y="revenue", title="Category Revenue")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Top Products
        top_products_query = """
        SELECT
            p.product_name,
            SUM(o.quantity * o.unit_price * (1-o.discount)) AS revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.product_id
        ORDER BY revenue DESC
        LIMIT 10
        """
        top_products = pd.read_sql(top_products_query, conn)
        st.subheader("Top Products By Revenue")
        st.dataframe(top_products, use_container_width=True)

# ====================================================
# CHURN PREDICTION PAGE
# ====================================================
elif page == "Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        total_orders = st.number_input("Total Orders", min_value=0, value=10)
        total_spending = st.number_input("Total Spending", min_value=0.0, value=5000.0)
        avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=500.0)
        days_since_last_order = st.number_input("Days Since Last Order", min_value=0, value=15)

    with col2:
        return_rate = st.number_input("Return Rate", min_value=0.0, max_value=1.0, value=0.10)
        avg_delivery_days = st.number_input("Average Delivery Days", min_value=0.0, value=4.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        membership_type = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold", "Premium"])

    if st.button("Predict Churn", type="primary"):
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

        st.subheader("Input Customer Profile")
        st.dataframe(input_df, use_container_width=True)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Churn Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'red'},
                    'steps': [
                        {'range': [0, 40], 'color': 'lightgreen'},
                        {'range': [40, 70], 'color': 'orange'},
                        {'range': [70, 100], 'color': 'red'}
                    ]
                }
            )
        )

        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **Customer Likely To Churn**\n\nChurn Probability: **{probability:.2%}**")
        else:
            st.success(f"✅ **Customer Likely To Stay**\n\nRetention Probability: **{(1-probability):.2%}**")

        if probability >= 0.75:
            st.error("🚨 High Risk Customer")
        elif probability >= 0.50:
            st.warning("⚠️ Medium Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

        numeric_cols = [
            "total_orders",
            "total_spending",
            "avg_order_value",
            "days_since_last_order",
            "return_rate",
            "avg_delivery_days",
            "age"
        ]

        distances = euclidean_distances(
            customer_features_df[numeric_cols],
            input_df[numeric_cols]
        )

        closest_idx = distances.argmin()
        closest_customer = customer_features_df.iloc[closest_idx]

        st.subheader("Most Similar Historical Customer")
        st.dataframe(closest_customer.to_frame().T, use_container_width=True)

        customer_features_df["distance"] = distances
        top5 = customer_features_df.sort_values("distance").head(5)

        st.subheader("Top 5 Similar Customers")
        st.dataframe(top5, use_container_width=True)

# ====================================================
# SENTIMENT ANALYSIS PAGE
# ====================================================
elif page == "Sentiment Analysis":

    st.title("💬 Customer Sentiment Analysis")
    st.write("Enter a customer review below to classify its sentiment.")

    user_review = st.text_area("Review Text", placeholder="e.g., terrible product, worst purchase ever...")

    if st.button("Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review to analyze.")
        else:
            cleaned_review = clean_text(user_review)
            transformed_vector = tfidf_vectorizer.transform([cleaned_review])

            prediction = sentiment_model.predict(transformed_vector)[0]
            probabilities = sentiment_model.predict_proba(transformed_vector)[0]
            classes = sentiment_model.classes_

            st.subheader("Analysis Result")

            if prediction == "Positive":
                st.success(f"**Predicted Sentiment:** {prediction}")
            elif prediction == "Negative":
                st.error(f"**Predicted Sentiment:** {prediction}")
            else:
                st.info(f"**Predicted Sentiment:** {prediction}")

            prob_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities
            })

            sentiment_fig = px.bar(
                prob_df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                text="Probability",
                title="Prediction Confidence"
            )

            st.plotly_chart(sentiment_fig, use_container_width=True)

            confidence = max(probabilities)
            st.progress(float(confidence))
            st.write(f"Model Confidence: {confidence:.2%}")

            with st.expander("Detailed Model Analysis"):
                st.write(f"**Processed Text:** `{cleaned_review}`")
                st.write("**Confidence Breakdown:**")
                for cls, prob in zip(classes, probabilities):
                    st.write(f"- **{cls}:** {prob*100:.2f}%")