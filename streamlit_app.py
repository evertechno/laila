import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Ever AI - Data Extraction & Analysis")
st.write("Use generative AI to analyze data and get insights based on your prompt.")

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
    
    # Correlation matrix for numerical columns
    if data.select_dtypes(include=['number']).shape[1] > 1:
        st.write("### Correlation Matrix:")
        corr = data.corr()
        st.write(corr)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Data cleaning options
    st.write("### Data Cleaning Options:")
    if st.checkbox("Remove missing values"):
        data = data.dropna()
        st.write("Missing values dropped.")
    if st.checkbox("Remove duplicates"):
        data = data.drop_duplicates()
        st.write("Duplicates removed.")
    
    # Numeric and Categorical Column Split
    st.write("### Numeric and Categorical Columns:")
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(exclude=['number']).columns
    st.write(f"Numeric Columns: {list(numeric_columns)}")
    st.write(f"Categorical Columns: {list(categorical_columns)}")

    # AI-based Column Interpretation
    if st.button("Interpret Columns with AI"):
        prompt = f"Analyze this dataset and determine the type of each column (numeric, categorical). Data: {data.to_dict()}"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        st.write("AI Interpretation:")
        st.write(response.text)

    # Data Sampling
    st.write("### Data Sampling Options:")
    if st.checkbox("Random Sampling"):
        sample_size = st.slider("Select sample size:", 1, data.shape[0], 100)
        sampled_data = data.sample(sample_size)
        st.write("Sampled Data:")
        st.write(sampled_data)

    # Data Normalization/Standardization
    st.write("### Data Normalization/Standardization:")
    scaler_option = st.radio("Choose transformation:", ('None', 'Standardize (Z-Score)', 'Normalize (Min-Max)'))
    if scaler_option == 'Standardize (Z-Score)':
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        st.write("Data standardized.")
    elif scaler_option == 'Normalize (Min-Max)':
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        st.write("Data normalized.")

    # Basic Visualization
    st.write("### Data Visualizations:")
    chart_type = st.selectbox("Select chart type:", ["Bar", "Line", "Histogram", "Scatter", "Heatmap"])
    
    if chart_type == "Bar":
        col = st.selectbox("Choose column for bar chart:", data.columns)
        fig, ax = plt.subplots()
        data[col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    elif chart_type == "Line":
        col = st.selectbox("Choose column for line chart:", data.columns)
        fig, ax = plt.subplots()
        data[col].plot(kind='line', ax=ax)
        st.pyplot(fig)
    elif chart_type == "Histogram":
        col = st.selectbox("Choose column for histogram:", data.columns)
        fig, ax = plt.subplots()
        data[col].plot(kind='hist', ax=ax)
        st.pyplot(fig)
    elif chart_type == "Scatter":
        x_col = st.selectbox("Choose X-axis column:", data.columns)
        y_col = st.selectbox("Choose Y-axis column:", data.columns)
        fig, ax = plt.subplots()
        data.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # User Custom Analysis Prompt
    st.write("### Ask AI to Analyze the Data:")
    ai_prompt = st.text_input("Enter a custom prompt for AI:", "What insights can you gather from this data?")
    if ai_prompt and st.button("Get AI Analysis"):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Analyze the following dataset and provide insights. Data: {data.to_dict()}. Prompt: {ai_prompt}"
            response = model.generate_content(prompt)
            st.write("AI's Analysis:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display Outliers
    st.write("### Outlier Detection:")
    if st.checkbox("Show outliers in numeric columns"):
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
            st.write(f"Outliers for column {col}:")
            st.write(outliers)

    # Dataset Download
    st.write("### Download Processed Data:")
    if st.button("Download Processed Data as CSV"):
        csv = data.to_csv(index=False)
        st.download_button("Download", csv, file_name="processed_data.csv", mime="text/csv")

    # Summary Report Generation via AI
    if st.checkbox("Generate AI Summary Report"):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Generate a summary report for this dataset. Data: {data.to_dict()}"
            response = model.generate_content(prompt)
            st.write("AI Report:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error generating AI report: {e}")

# General Prompt Input for AI
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
