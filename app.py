# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Set a title for the dashboard
st.title("Beginner's Guide to Data Analysis and Machine Learning")

# Sidebar for navigating between different tutorial modules
curriculum_sections = [
    "Introduction to Data Analysis",
    "Data Visualization",
    "Simple Linear Regression",
    "Decision Tree Regression"
]
selected_section = st.sidebar.selectbox("Choose a Module", curriculum_sections)

# File uploader for users to upload their dataset once
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

# Module 1: Introduction to Data Analysis
if selected_section == "Introduction to Data Analysis":
    st.header("Introduction to Data Analysis")
    st.write("In this module, we'll start with the basics, such as exploring and summarizing your data.")
    
    if uploaded_file:
        # Option to display summary statistics
        if st.checkbox("Show Data Summary (Statistics)"):
            st.write("This summary includes statistics like mean, standard deviation, and percentiles.")
            st.write(data.describe())

        # Option to show data types of each column
        if st.checkbox("Show Data Types"):
            st.write("This shows the type of data in each column (e.g., integer, float, object).")
            st.write(data.dtypes)
        
        # Example code snippet for the above tasks
        st.code("""
# Displaying basic statistics and data types
def summarize_data(data):
    print("Data Summary:")
    print(data.describe())
    
    print("\nData Types:")
    print(data.dtypes)

# Usage
summarize_data(data)
""", language='python')

# Module 2: Data Visualization
elif selected_section == "Data Visualization":
    st.header("Data Visualization")
    st.write("Learn to create simple visualizations that make data easier to understand.")
    
    if uploaded_file:
        # User selects columns to plot on X and Y axes
        x_column = st.selectbox("Choose X-axis", data.columns)
        y_column = st.selectbox("Choose Y-axis", data.columns)
        
        # Plot line chart
        if st.button("Generate Line Chart"):
            st.line_chart(data[[x_column, y_column]].set_index(x_column))
        
        # Sample code for visualization
        st.code("""
import matplotlib.pyplot as plt

# Function to plot a line chart
def plot_line_chart(data, x_column, y_column):
    data.plot(x=x_column, y=y_column, kind='line')
    plt.title("Line Chart")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

# Usage
plot_line_chart(data, 'column_name_x', 'column_name_y')
""", language='python')

# Module 3: Simple Linear Regression
elif selected_section == "Simple Linear Regression":
    st.header("Simple Linear Regression")
    st.write("Learn how to create a basic Linear Regression model to predict a target variable.")
    
    if uploaded_file:
        # Choose target and features
        target_column = st.selectbox("Choose Target (Y) Column", data.columns)
        feature_columns = st.multiselect("Choose Feature (X) Columns", [col for col in data.columns if col != target_column])
        
        if st.button("Train Linear Regression Model") and feature_columns:
            X = data[feature_columns]
            y = data[target_column]
            model = LinearRegression()
            model.fit(X, y)  # Train the model
            predictions = model.predict(X)  # Predict using the model
            mse = mean_squared_error(y, predictions)  # Calculate error
            st.write(f"Mean Squared Error: {mse:.4f}")
        
        # Example code for Linear Regression
        st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features (X) and target (y)
X = data[feature_columns]
y = data[target_column]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions and calculate error
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)
""", language='python')

# Module 4: Decision Tree Regression
elif selected_section == "Decision Tree Regression":
    st.header("Decision Tree Regression")
    st.write("Use a Decision Tree model, which is more complex than linear regression and can capture non-linear patterns.")
    
    if uploaded_file:
        target_column = st.selectbox("Choose Target (Y) Column", data.columns)
        feature_columns = st.multiselect("Choose Feature (X) Columns", [col for col in data.columns if col != target_column])
        
        if st.button("Train Decision Tree Model") and feature_columns:
            X = data[feature_columns]
            y = data[target_column]
            model = DecisionTreeRegressor()  # Initialize model
            model.fit(X, y)  # Train the model
            predictions = model.predict(X)  # Predict using the model
            mse = mean_squared_error(y, predictions)  # Calculate error
            st.write(f"Decision Tree Mean Squared Error: {mse:.4f}")
        
        # Example code for Decision Tree Regression
        st.code("""
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Define features (X) and target (y)
X = data[feature_columns]
y = data[target_column]

# Train the Decision Tree model
model = DecisionTreeRegressor()
model.fit(X, y)

# Make predictions and calculate error
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
print("Decision Tree Mean Squared Error:", mse)
""", language='python')
