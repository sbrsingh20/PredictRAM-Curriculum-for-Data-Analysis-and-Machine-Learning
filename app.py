import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set the title for the Streamlit app
st.title("Data Analysis and Machine Learning Interactive Curriculum")

# Sidebar navigation for curriculum modules
modules = [
    "Data Upload and Overview",
    "Data Cleaning and Preprocessing",
    "Feature Engineering",
    "Linear Regression Model",
    "Decision Tree Model",
    "Model Evaluation and Visualization",
    "Scenario Analysis"
]
selected_module = st.sidebar.selectbox("Select a Module", modules)

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

# Module 1: Data Overview
if selected_module == "Data Upload and Overview":
    st.header("Module 1: Data Overview")
    if uploaded_file:
        st.write("Data Overview:")
        st.write("Data Shape:", data.shape)
        st.write("Data Info:")
        st.write(data.info())
        st.write("Data Statistics:")
        st.write(data.describe())
    else:
        st.write("Please upload a CSV file to continue.")

# Module 2: Data Cleaning and Preprocessing
elif selected_module == "Data Cleaning and Preprocessing":
    st.header("Module 2: Data Cleaning and Preprocessing")
    if uploaded_file:
        st.write("Checking for Missing Values:")
        st.write(data.isnull().sum())
        if st.checkbox("Fill Missing Values (Forward Fill)"):
            data.fillna(method='ffill', inplace=True)
            st.write("Missing values have been filled.")
        if st.checkbox("Drop Duplicate Rows"):
            data.drop_duplicates(inplace=True)
            st.write("Duplicate rows have been removed.")
        st.write("Processed Data Preview:")
        st.dataframe(data.head())
    else:
        st.write("Please upload a CSV file to continue.")

# Module 3: Feature Engineering
elif selected_module == "Feature Engineering":
    st.header("Module 3: Feature Engineering")
    if uploaded_file:
        st.write("Feature Engineering Options:")
        selected_column = st.selectbox("Select Column for Moving Average", data.columns)
        window_size = st.slider("Select Window Size", min_value=2, max_value=10, value=5)
        data[f"{selected_column}_Moving_Avg"] = data[selected_column].rolling(window=window_size).mean()
        st.write("Data with Moving Average:")
        st.dataframe(data[[selected_column, f"{selected_column}_Moving_Avg"]].head(10))
    else:
        st.write("Please upload a CSV file to continue.")

# Module 4: Linear Regression Model
elif selected_module == "Linear Regression Model":
    st.header("Module 4: Linear Regression Model")
    if uploaded_file:
        target = st.selectbox("Select Target Variable", data.columns)
        features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])
        
        if st.button("Train Linear Regression Model"):
            X = data[features].dropna()
            y = data[target].loc[X.index]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write("Linear Regression Model Trained.")
            st.write(f"Mean Squared Error: {mse:.4f}")
    else:
        st.write("Please upload a CSV file to continue.")

# Module 5: Decision Tree Model
elif selected_module == "Decision Tree Model":
    st.header("Module 5: Decision Tree Model")
    if uploaded_file:
        target = st.selectbox("Select Target Variable", data.columns)
        features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

        if st.button("Train Decision Tree Model"):
            X = data[features].dropna()
            y = data[target].loc[X.index]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write("Decision Tree Model Trained.")
            st.write(f"Mean Squared Error: {mse:.4f}")
    else:
        st.write("Please upload a CSV file to continue.")

# Module 6: Model Evaluation and Visualization
elif selected_module == "Model Evaluation and Visualization":
    st.header("Module 6: Model Evaluation and Visualization")
    if uploaded_file:
        target = st.selectbox("Select Target Variable", data.columns)
        features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

        if st.button("Evaluate and Visualize"):
            X = data[features].dropna()
            y = data[target].loc[X.index]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Linear Regression
            lin_model = LinearRegression()
            lin_model.fit(X_train, y_train)
            lin_predictions = lin_model.predict(X_test)
            lin_mse = mean_squared_error(y_test, lin_predictions)
            
            # Decision Tree
            tree_model = DecisionTreeRegressor()
            tree_model.fit(X_train, y_train)
            tree_predictions = tree_model.predict(X_test)
            tree_mse = mean_squared_error(y_test, tree_predictions)
            
            # Results
            st.write(f"Linear Regression MSE: {lin_mse:.4f}")
            st.write(f"Decision Tree MSE: {tree_mse:.4f}")
            
            # Plotting
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label="Actual")
            ax.plot(lin_predictions, label="Linear Regression Predictions")
            ax.plot(tree_predictions, label="Decision Tree Predictions")
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to continue.")

# Module 7: Scenario Analysis
elif selected_module == "Scenario Analysis":
    st.header("Module 7: Scenario Analysis")
    if uploaded_file:
        target = st.selectbox("Select Target Variable", data.columns)
        features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

        if st.button("Run Scenario Analysis"):
            X = data[features].dropna()
            y = data[target].loc[X.index]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            modified_X_test = X_test.copy()
            feature_to_modify = st.selectbox("Select Feature to Modify", features)
            modification_amount = st.slider("Modification Amount", -10, 10, 1)
            modified_X_test[feature_to_modify] += modification_amount

            modified_predictions = model.predict(modified_X_test)
            fig, ax = plt.subplots()
            ax.plot(y_test.values, label="Actual")
            ax.plot(modified_predictions, label="Modified Predictions")
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to continue.")
