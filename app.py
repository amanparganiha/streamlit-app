import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Credit Rating Dashboard", layout="wide")

# Title and description
st.title("Credit Rating Prediction Dashboard")
st.write("""
This interactive dashboard allows exploration of credit rating data combined with financial indicators 
and NLP-derived sentiment scores. Upload a CSV file with credit rating data, or use the sample data. 
Use the sidebar to filter the data and customize visualizations.
""")

# Sidebar - Data Upload
st.sidebar.header("Upload Data")
data_file = st.sidebar.file_uploader("Upload credit_ratings_multimodal_final.csv", type=["csv"])
if data_file is not None:
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    # Generate sample placeholder data if no file uploaded
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        'investment_grade': np.random.choice(['Investment', 'Non-Investment'], size=n, p=[0.7, 0.3]),
        'rating': np.random.choice(['AAA','AA','A','BBB','BB','B'], size=n),
        'sector': np.random.choice(['Technology','Healthcare','Finance','Energy'], size=n),
        'financial_score': np.random.normal(50, 15, size=n).clip(0),
        'sentiment': np.random.normal(0, 0.3, size=n),
        'asset_quality': np.random.normal(100, 20, size=n).clip(0),
        'leverage': np.random.normal(2, 0.5, size=n).clip(0),
        'coverage_ratio': np.random.normal(5, 1.5, size=n).clip(0)
    })
    st.info("Using sample data. Upload a CSV to use your own dataset.")

# Sidebar - Optional Feature Importance Upload
feat_file = st.sidebar.file_uploader("Upload feature_importance.csv (Optional)", type=["csv"])

# Sidebar - Filters
st.sidebar.header("Filters")
# Filter by Investment Grade
if 'investment_grade' in df.columns:
    inv_grades = df['investment_grade'].unique().tolist()
    inv_grades.sort()
    selected_inv = st.sidebar.multiselect("Investment Grade", inv_grades, default=inv_grades)
    if selected_inv:
        df = df[df['investment_grade'].isin(selected_inv)]
# Filter by Rating
if 'rating' in df.columns:
    ratings = df['rating'].unique().tolist()
    ratings.sort()
    selected_ratings = st.sidebar.multiselect("Rating", ratings, default=ratings)
    if selected_ratings:
        df = df[df['rating'].isin(selected_ratings)]
# Filter by Sector
if 'sector' in df.columns:
    sectors = df['sector'].unique().tolist()
    sectors.sort()
    selected_sectors = st.sidebar.multiselect("Sector", sectors, default=sectors)
    if selected_sectors:
        df = df[df['sector'].isin(selected_sectors)]

# Main - Data Preview and Summary
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Summary Statistics")
st.write(df.describe(include='all').transpose())

# Visualizations
st.subheader("Visualizations")

# 1. Credit Rating Distribution
if 'rating' in df.columns:
    rating_counts = df['rating'].value_counts().reset_index()
    rating_counts.columns = ['rating', 'count']
    fig_rating = px.bar(rating_counts, x='rating', y='count', 
                        title="Distribution of Credit Ratings",
                        labels={'count': 'Number of Records'})
    st.plotly_chart(fig_rating, use_container_width=True)

# 2. Financial Score vs Sector
if 'financial_score' in df.columns and 'sector' in df.columns:
    fig_fin_sec = px.box(df, x='sector', y='financial_score', 
                         title="Financial Score by Sector", 
                         labels={'financial_score': 'Financial Score'})
    st.plotly_chart(fig_fin_sec, use_container_width=True)

# 3. Sentiment vs Credit Rating
if 'sentiment' in df.columns and 'rating' in df.columns:
    fig_sent_rating = px.box(df, x='rating', y='sentiment', 
                             title="NLP Sentiment by Credit Rating",
                             labels={'sentiment': 'Sentiment Score'})
    st.plotly_chart(fig_sent_rating, use_container_width=True)

# Interactive Custom Charts
st.subheader("Custom Chart Explorer")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    # Histogram
    hist_col = st.selectbox("Select a numeric column for histogram", options=numeric_cols)
    fig_hist = px.histogram(df, x=hist_col, title=f"Histogram of {hist_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter plot
    if len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X-axis", options=numeric_cols, index=0, key="xaxis")
        y_col = col2.selectbox("Y-axis", options=numeric_cols, index=1, key="yaxis")
        fig_scatter = px.scatter(df, x=x_col, y=y_col, 
                                 title=f"{y_col} vs {x_col}", 
                                 labels={x_col: x_col, y_col: y_col})
        st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.write("No numeric columns available for custom charts.")

# Feature Importance Visualization (Optional)
if feat_file is not None:
    st.subheader("Feature Importances")
    try:
        df_feat = pd.read_csv(feat_file)
        if 'feature' in df_feat.columns and 'importance' in df_feat.columns:
            df_feat = df_feat.sort_values(by='importance', ascending=False).head(10)
            fig_feat = px.bar(df_feat, x='importance', y='feature', orientation='h',
                              title="Top Feature Importances", labels={'importance': 'Importance', 'feature': 'Feature'})
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.error("The uploaded file should contain 'feature' and 'importance' columns.")
    except Exception as e:
        st.error(f"Error loading feature importance data: {e}")
