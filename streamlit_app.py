import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, mean_squared_error, mean_absolute_error
from wordcloud import WordCloud
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from scipy.stats import ttest_ind
from io import BytesIO
import google.generativeai as genai

# Configure the API key for Gemini AI
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Advanced Data Analysis and AI Insights")
st.write("An interactive platform for data analysis with AI-powered insights.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display data preview
    st.write("### Data Preview")
    st.write(data.head())

    # Data Filtering
    st.write("### Data Filtering:")
    column_filter = st.selectbox("Choose a column to filter:", data.columns)
    filter_value = st.text_input(f"Filter rows where {column_filter} is:", "")
    if filter_value:
        filtered_data = data[data[column_filter] == filter_value]
        st.write(filtered_data)

    # Box-Cox Transformation
    st.write("### Box-Cox Transformation:")
    if st.checkbox("Apply Box-Cox Transformation"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col], _ = stats.boxcox(data[col] + 1)  # Box-Cox transformation (handle zeros)
        st.write(data.head())

    # Data Normalization/Standardization
    st.write("### Data Normalization/Standardization:")
    normalize_method = st.radio("Choose method:", ('Min-Max', 'Z-Score'))
    if normalize_method == 'Min-Max':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled_data = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[np.number])))
    st.write(scaled_data)

    # Anomaly Detection using Isolation Forest
    st.write("### Anomaly Detection:")
    if st.checkbox("Detect Anomalies using Isolation Forest"):
        clf = IsolationForest()
        data['Anomaly'] = clf.fit_predict(data.select_dtypes(include=[np.number]))
        st.write(data[data['Anomaly'] == -1])  # Show detected anomalies

    # Word Cloud for Text Data
    st.write("### Word Cloud:")
    if st.checkbox("Generate Word Cloud"):
        text_column = st.selectbox("Select text column for word cloud:", data.select_dtypes(include=['object']).columns)
        wordcloud = WordCloud().generate(" ".join(data[text_column].dropna()))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

    # ROC Curve for Classification Models
    st.write("### ROC Curve:")
    if st.checkbox("Generate ROC Curve for Classification Model"):
        if 'target' in data.columns:  # Check if target column exists
            X = data.drop(columns=["target"])
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            st.pyplot()
        else:
            st.error("The target column is not available in the dataset.")

    # Feature Selection (Recursive Feature Elimination)
    st.write("### Feature Selection:")
    if st.checkbox("Apply Recursive Feature Elimination"):
        if 'target' in data.columns:
            X = data.drop(columns=["target"])
            y = data["target"]
            selector = RFE(LogisticRegression(), n_features_to_select=5)
            selector = selector.fit(X, y)
            st.write(f"Selected Features: {X.columns[selector.support_]}")
        else:
            st.error("The target column is not available in the dataset.")

    # PCA for Dimensionality Reduction
    st.write("### PCA for Dimensionality Reduction:")
    if st.checkbox("Apply PCA"):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(data.select_dtypes(include=[np.number]))
        pca_df = pd.DataFrame(principalComponents, columns=["Principal Component 1", "Principal Component 2"])
        st.write(pca_df)

    # Clustering (KMeans)
    st.write("### Clustering (KMeans):")
    if st.checkbox("Apply KMeans Clustering"):
        k = st.slider("Select number of clusters", min_value=2, max_value=10)
        kmeans = KMeans(n_clusters=k)
        data['Cluster'] = kmeans.fit_predict(data.select_dtypes(include=[np.number]))
        st.write(data.head())

    # Correlation Analysis
    st.write("### Correlation Analysis:")
    if st.checkbox("Show Correlation Heatmap"):
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()

    # Statistical T-test
    st.write("### Statistical Tests (T-Test):")
    if st.checkbox("Perform T-Test"):
        sample1_column = st.selectbox("Select first sample column:", data.columns)
        sample2_column = st.selectbox("Select second sample column:", data.columns)
        sample1 = data[sample1_column].dropna()
        sample2 = data[sample2_column].dropna()
        t_stat, p_val = ttest_ind(sample1, sample2)
        st.write(f"T-Statistic: {t_stat}, P-Value: {p_val}")

    # Predictive Model (Linear Regression)
    st.write("### Predictive Model:")
    if st.checkbox("Run Linear Regression"):
        if 'target' in data.columns:
            X = data.drop(columns=["target"])
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
        else:
            st.error("The target column is not available in the dataset.")

    # Data Imputation
    st.write("### Data Imputation:")
    if st.checkbox("Apply Imputation for Missing Values"):
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(data.select_dtypes(include=[np.number]))
        st.write(pd.DataFrame(imputed_data, columns=data.select_dtypes(include=[np.number]).columns))

    # Exporting Visualizations
    st.write("### Export Visualizations:")
    if st.checkbox("Export Visualization as PNG"):
        fig, ax = plt.subplots()
        sns.boxplot(data=data.select_dtypes(include=[np.number]), ax=ax)
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button("Download PNG", buf, "boxplot.png")

    # Chat Interface for AI Insights
    st.write("### AI Chat Interface for Data Insights:")
    chat_prompt = st.text_input("Ask AI to analyze or explain the data:", "")
    if chat_prompt:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Analyze the following dataset and answer: {chat_prompt}. Data: {data.to_dict()}"
        response = model.generate_content(prompt)
        st.write(response.text)

    # General AI Prompt
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
