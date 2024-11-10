import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Title of the Dashboard
st.title("Data Analysis and Machine Learning Curriculum Dashboard")

# Sidebar Navigation for Modules
modules = [
    "Introduction to Data Analysis",
    "Data Visualization",
    "Basic Linear Regression",
    "Advanced Model Training"
]
selected_module = st.sidebar.selectbox("Choose a Module", modules)

# Uploading data
st.sidebar.subheader("Upload your CSV data file")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
data = None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

# Module 1: Introduction to Data Analysis
if selected_module == "Introduction to Data Analysis":
    st.header("Introduction to Data Analysis")
    st.write("In this module, we explore data structure and summary statistics.")

    if data is not None:
        # Show data summary
        if st.checkbox("Show Data Summary"):
            st.write(data.describe())
        
        # Show data types and missing values
        if st.checkbox("Show Data Types and Missing Values"):
            st.write(data.dtypes)
            st.write("Missing Values per Column:")
            st.write(data.isnull().sum())

# Module 2: Data Visualization
elif selected_module == "Data Visualization":
    st.header("Data Visualization")
    st.write("Visualize relationships between different variables.")

    if data is not None:
        x_column = st.selectbox("Choose X-axis for plotting", data.columns)
        y_column = st.selectbox("Choose Y-axis for plotting", data.columns)

        if st.button("Generate Plot"):
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=data[x_column], y=data[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"Line Plot of {y_column} over {x_column}")
            st.pyplot(plt)

# Module 3: Basic Linear Regression
elif selected_module == "Basic Linear Regression":
    st.header("Basic Linear Regression")
    st.write("Train a linear regression model to predict one variable from others.")

    if data is not None:
        target_column = st.selectbox("Select Target Column", data.columns)
        feature_columns = st.multiselect("Select Feature Columns", [col for col in data.columns if col != target_column])

        if st.button("Train Linear Regression Model"):
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            st.write(f"Linear Regression Model Mean Squared Error: {mse:.4f}")

            # Option to save model
            if st.button("Save Model"):
                joblib.dump(model, 'linear_regression_model.joblib')
                st.write("Model saved as `linear_regression_model.joblib`.")

# Module 4: Advanced Model Training
elif selected_module == "Advanced Model Training":
    st.header("Advanced Model Training")
    st.write("Train a more complex model (Decision Tree Regressor) and compare performance.")

    if data is not None:
        target_column = st.selectbox("Select Target Column", data.columns)
        feature_columns = st.multiselect("Select Feature Columns", [col for col in data.columns if col != target_column])
        model_type = st.selectbox("Select Model Type", ["Decision Tree"])

        if st.button("Train Model"):
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "Decision Tree":
                model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            st.write(f"{model_type} Model Mean Squared Error: {mse:.4f}")

            # Option to save model
            if st.button("Save Model"):
                joblib.dump(model, f"{model_type.lower()}_model.joblib")
                st.write(f"Model saved as `{model_type.lower()}_model.joblib`.")
