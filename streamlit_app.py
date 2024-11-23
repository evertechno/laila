import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
from io import BytesIO
from textblob import TextBlob
import datetime

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Ever AI - Advanced Data Extraction & Analysis")
st.write("Use generative AI to analyze data and gain actionable insights.")

# Sidebar for File Upload
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV/Excel/JSON/Parquet file", type=["csv", "xlsx", "json", "parquet"])

# Function to read data based on file type
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    elif file.name.endswith('.parquet'):
        return pd.read_parquet(file)
    else:
        return None

# Placeholder for file data and analysis options
if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    # Display basic data info
    st.write("### Uploaded Data Preview:")
    st.write(data.head())

    # Data Information & Exploration
    st.write("### Data Overview:")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write("Data Types:")
    st.write(data.dtypes)
    
    # Missing values summary
    st.write("### Missing Values:")
    st.write(data.isnull().sum())

    # Missing Data Imputation (Mean/Median)
    st.write("### Impute Missing Data:")
    if st.checkbox("Impute Missing Data"):
        strategy = st.radio("Choose Imputation Strategy:", ['Mean', 'Median', 'Mode'])
        if strategy == 'Mean':
            data = data.fillna(data.mean())
        elif strategy == 'Median':
            data = data.fillna(data.median())
        else:
            data = data.fillna(data.mode().iloc[0])
        st.write("Missing data imputed.")

    # Advanced Outlier Detection (Z-score method)
    st.write("### Outlier Detection:")
    if st.checkbox("Detect Outliers using Z-Score"):
        threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0)
        z_scores = np.abs((data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std())
        outliers = (z_scores > threshold).sum()
        st.write(f"Number of outliers: {outliers.sum()}")

    # Feature Engineering: Polynomial Features (e.g., creating interaction terms)
    st.write("### Feature Engineering:")
    if st.checkbox("Generate Polynomial Features"):
        degree = st.slider("Polynomial Degree:", 2, 5, 2)
        poly = pd.DataFrame(StandardScaler().fit_transform(data[numeric_columns]) ** degree)
        st.write("Polynomial features generated.")
    
    # Handling Categorical Data (One-Hot Encoding)
    st.write("### One-Hot Encoding for Categorical Columns:")
    if st.checkbox("One-Hot Encode Categorical Columns"):
        categorical_columns = data.select_dtypes(include=['object']).columns
        encoded_data = pd.get_dummies(data, columns=categorical_columns)
        st.write(f"One-Hot Encoded Data: {encoded_data.head()}")

    # Advanced Data Sampling: Stratified Sampling
    st.write("### Stratified Sampling:")
    if st.checkbox("Stratified Sampling (by target)"):
        stratify_col = st.selectbox("Select column to stratify by:", data.columns)
        sample_size = st.slider("Select sample size:", 1, data.shape[0], 100)
        stratified_data = data.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(sample_size))
        st.write(f"Stratified Sample: {stratified_data.head()}")

    # K-Means Clustering
    st.write("### K-Means Clustering:")
    if st.checkbox("Perform K-Means Clustering"):
        n_clusters = st.slider("Number of clusters:", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters)
        data['Cluster'] = kmeans.fit_predict(data[numeric_columns])
        st.write("Clusters assigned to data.")

    # Principal Component Analysis (PCA)
    st.write("### Principal Component Analysis (PCA):")
    if st.checkbox("Apply PCA"):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data[numeric_columns])
        st.write("PCA Components:")
        st.write(pd.DataFrame(pca_result, columns=['PCA1', 'PCA2']))

    # Time-Series Analysis (if date column exists)
    st.write("### Time-Series Analysis:")
    if st.checkbox("Analyze Time-Series Data"):
        date_column = st.selectbox("Select date column for time-series analysis:", data.columns)
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        st.line_chart(data)

    # Model Training Interface (e.g., Linear Regression)
    st.write("### Model Training Interface:")
    if st.checkbox("Train Simple ML Model (Linear Regression)"):
        target_column = st.selectbox("Select target column:", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f"Model Performance:\nMSE: {mse}\nR²: {r2}")

    # Prediction and Forecasting
    st.write("### Prediction and Forecasting:")
    if st.checkbox("Make Predictions using Linear Regression"):
        target_column = st.selectbox("Choose target variable for prediction:", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        model = LinearRegression()
        model.fit(X, y)
        st.write("Model trained. Make a prediction.")
        sample_input = st.text_area("Input data for prediction (as JSON):")
        if sample_input:
            sample_input = eval(sample_input)
            prediction = model.predict(pd.DataFrame([sample_input]))
            st.write(f"Prediction: {prediction}")

    # Data Aggregation (GroupBy, Pivot Table)
    st.write("### Data Aggregation:")
    if st.checkbox("GroupBy Aggregation"):
        group_column = st.selectbox("Select column to group by:", data.columns)
        aggregation = data.groupby(group_column).mean()
        st.write(f"Aggregated Data:\n{aggregation}")

    # Statistical Tests: T-Test
    st.write("### Perform Statistical Tests:")
    if st.checkbox("Perform T-Test"):
        sample1 = st.selectbox("Select first sample column:", data.columns)
        sample2 = st.selectbox("Select second sample column:", data.columns)
        t_stat, p_val = st.t_test(data[sample1], data[sample2])
        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")

    # Text Analysis (Sentiment Analysis using TextBlob)
    st.write("### Text Analysis:")
    if st.checkbox("Perform Sentiment Analysis"):
        text_column = st.selectbox("Choose text column for sentiment analysis:", data.columns)
        sentiment = data[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        st.write(f"Sentiment Analysis Result:\n{sentiment.describe()}")

    # Export Plots as Images/PDF
    st.write("### Export Visualizations:")
    if st.checkbox("Export Visualization as PNG"):
        fig, ax = plt.subplots()
        data[numeric_columns].plot(kind='box', ax=ax)
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download PNG", buf, "plot.png")

    # File Version Control (Backup)
    st.write("### File Version Control:")
    if st.checkbox("Enable Version Control"):
        file_backup = st.file_uploader("Upload previous version to compare with current data", type=["csv", "xlsx"])
        if file_backup is not None:
            backup_data = load_data(file_backup)
            diff_data = pd.concat([backup_data, data]).drop_duplicates(keep=False)
            st.write(f"Changes in Data:\n{diff_data}")
    
    # AI Chat Interface for Data Analysis
    st.write("### AI Chat Interface for Data Analysis:")
    chat_prompt = st.text_input("Ask AI to analyze or explain the data:", "")
    if chat_prompt:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Analyze the following dataset and answer: {chat_prompt}. Data: {data.to_dict()}"
        response = model.generate_content(prompt)
        st.write(response.text)

# General Prompt Input for AI (default interface)
st.write("### General AI Prompt")
user_prompt = st.text_input("Enter your general prompt for AI:", "Best alternatives to JavaScript?")
if st.button("Generate Response"):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(user_prompt)
        st.write("Response:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Error: {e}")
