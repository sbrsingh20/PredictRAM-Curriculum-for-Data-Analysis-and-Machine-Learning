# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Set a title for the dashboard
st.title("Interactive Curriculum for Data Analysis and Machine Learning")

# Sidebar Navigation for Curriculum Modules
curriculum_sections = [
    "Introduction to Data Analysis",
    "Data Visualization",
    "Basic Linear Regression",
    "Advanced Model Training"
]
selected_section = st.sidebar.selectbox("Choose a Module", curriculum_sections)

# Upload data once, accessible across all modules
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

# Module 1: Introduction to Data Analysis
if selected_section == "Introduction to Data Analysis":
    st.header("Introduction to Data Analysis")
    st.write("This module provides an overview of basic data analysis techniques.")
    
    if uploaded_file:
        # Display basic statistics
        if st.checkbox("Show Data Summary"):
            st.write(data.describe())

        # Show data types
        if st.checkbox("Show Data Types"):
            st.write(data.dtypes)
        
        # Example code snippet
        st.code("""
# Code snippet for data summary
def summarize_data(data):
    return data.describe()

# Usage
summary = summarize_data(data)
print(summary)
""", language='python')

# Module 2: Data Visualization
elif selected_section == "Data Visualization":
    st.header("Data Visualization")
    st.write("Learn how to visualize data to gain insights.")
    
    if uploaded_file:
        # Select columns for plotting
        x_column = st.selectbox("Choose X-axis", data.columns)
        y_column = st.selectbox("Choose Y-axis", data.columns)
        
        # Plot data
        if st.button("Plot Data"):
            st.line_chart(data[[x_column, y_column]])
        
        # Example code snippet
        st.code("""
import matplotlib.pyplot as plt

# Function to plot data
def plot_data(data, x_column, y_column):
    data.plot(x=x_column, y=y_column, kind='line')
    plt.show()

# Usage
plot_data(data, 'X_column_name', 'Y_column_name')
""", language='python')

# Module 3: Basic Linear Regression
elif selected_section == "Basic Linear Regression":
    st.header("Basic Linear Regression")
    st.write("Train a linear regression model to predict a target variable.")
    
    if uploaded_file:
        # Select target column for prediction
        target_column = st.selectbox("Choose Target Column", data.columns)
        feature_columns = st.multiselect("Choose Feature Columns", [col for col in data.columns if col != target_column])
        
        if st.button("Train Linear Regression Model"):
            X = data[feature_columns]
            y = data[target_column]
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            st.write(f"Mean Squared Error: {mse:.4f}")
        
        # Example code snippet
        st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train a Linear Regression Model
X = data[feature_columns]
y = data[target_column]
model = LinearRegression()
model.fit(X, y)

# Predict and calculate error
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print("MSE:", mse)
""", language='python')

# Module 4: Advanced Model Training
elif selected_section == "Advanced Model Training":
    st.header("Advanced Model Training")
    st.write("This module covers more complex models like Decision Trees.")
    
    if uploaded_file:
        target_column = st.selectbox("Choose Target Column", data.columns)
        feature_columns = st.multiselect("Choose Feature Columns", [col for col in data.columns if col != target_column])
        
        # Select model type
        model_type = st.selectbox("Select Model Type", ["Decision Tree", "Random Forest"])
        
        if st.button("Train Model"):
            X = data[feature_columns]
            y = data[target_column]
            
            if model_type == "Decision Tree":
                model = DecisionTreeRegressor()
            # Add code for other model types if needed
            
            model.fit(X, y)
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            st.write(f"{model_type} Model Mean Squared Error: {mse:.4f}")
        
        # Example code snippet
        st.code("""
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Train a Decision Tree Model
X = data[feature_columns]
y = data[target_column]
model = DecisionTreeRegressor()
model.fit(X, y)

# Predict and calculate error
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print("Decision Tree MSE:", mse)
""", language='python')
