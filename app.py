import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title and sidebar
st.title("Stock Price and Income Statement Analysis")
st.sidebar.header("Upload Data Files")

# Upload stock price and income statement files
stock_file = st.sidebar.file_uploader("Upload Stock Price Data (CSV)", type="csv")
income_file = st.sidebar.file_uploader("Upload Income Statement Data (CSV)", type="csv")

if stock_file and income_file:
    # Load datasets
    stock_data = pd.read_csv(stock_file, parse_dates=['Date'])
    income_data = pd.read_csv(income_file, parse_dates=['Date'])
    
    # Ensure 'Date' columns are in datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    income_data['Date'] = pd.to_datetime(income_data['Date'], errors='coerce')

    # Display raw data preview
    st.write("### Stock Price Data")
    st.dataframe(stock_data.head())
    st.write("### Income Statement Data")
    st.dataframe(income_data.head())
    
    # Allow user to select time periods
    st.sidebar.subheader("Select Time Period")
    min_date = max(stock_data['Date'].min(), income_data['Date'].min())
    max_date = min(stock_data['Date'].max(), income_data['Date'].max())
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Filter data within the selected time range
    filtered_stock_data = stock_data[(stock_data['Date'] >= pd.to_datetime(start_date)) & 
                                     (stock_data['Date'] <= pd.to_datetime(end_date))]
    filtered_income_data = income_data[(income_data['Date'] >= pd.to_datetime(start_date)) & 
                                       (income_data['Date'] <= pd.to_datetime(end_date))]
    
    # Merge datasets on Date
    merged_data = pd.merge(filtered_stock_data, filtered_income_data, on='Date', how='inner')
    st.write("### Merged Data (Stock & Income Statement)")
    st.dataframe(merged_data.head())

    # Module 1: Correlation Analysis
    st.header("Module 1: Correlation Analysis")
    correlation_matrix = merged_data.corr()
    st.write("### Correlation Matrix")
    st.dataframe(correlation_matrix)
    
    # Heatmap for correlation
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Module 2: Regression Analysis
st.header("Module 2: Regression Analysis")
target_variable = st.selectbox("Select Target Variable for Stock Price Prediction", 
                               options=filtered_stock_data.columns)
feature_variable = st.selectbox("Select Feature Variable for Regression Analysis",
                                options=[col for col in filtered_income_data.columns if col != 'Date'])
    
if target_variable and feature_variable:
    # Ensure merged_data has no NaN values in the selected columns
    merged_data_clean = merged_data.dropna(subset=[feature_variable, target_variable])
    
    if len(merged_data_clean) < 2:  # Check if there is enough data
        st.warning("Not enough data available for regression. Please select a valid time range or metrics.")
    else:
        X = merged_data_clean[[feature_variable]]
        y = merged_data_clean[target_variable]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"### Regression Model Results: {target_variable} ~ {feature_variable}")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R-Squared: {r2:.4f}")
        
        # Plot actual vs predicted
        st.write("### Actual vs Predicted Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot(y_test, y_test, color='red')  # Line for perfect prediction
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{target_variable} Prediction vs Actual")
        st.pyplot(fig)


    # Module 3: Trend Analysis
    st.header("Module 3: Trend Analysis for Different Time Periods")
    st.write("### Select Metric and Time Period")
    trend_metric = st.selectbox("Choose Metric for Trend Analysis", options=merged_data.columns)
    trend_period = st.selectbox("Select Time Period (Monthly/Quarterly)", ["Monthly", "Quarterly"])

    if trend_period == "Monthly":
        trend_data = merged_data.set_index('Date').resample('M')[trend_metric].mean()
    else:
        trend_data = merged_data.set_index('Date').resample('Q')[trend_metric].mean()

    st.write(f"### {trend_period} Trend Analysis for {trend_metric}")
    fig, ax = plt.subplots()
    trend_data.plot(ax=ax)
    ax.set_title(f"{trend_metric} Trend ({trend_period})")
    ax.set_xlabel("Date")
    ax.set_ylabel(trend_metric)
    st.pyplot(fig)

    # Module 4: Scenario Analysis - Adjust Income Statement Metric
    st.header("Module 4: Scenario Analysis")
    scenario_metric = st.selectbox("Select Metric for Scenario Analysis", options=merged_data.columns)
    adjustment_factor = st.slider("Adjustment Factor (%)", -50, 50, 0)
    
    # Apply adjustment
    adjusted_data = merged_data.copy()
    adjusted_data[scenario_metric] = adjusted_data[scenario_metric] * (1 + adjustment_factor / 100)
    
    # Retrain with adjusted data if same metric selected
    if target_variable and feature_variable:
        X_adjusted = adjusted_data[[feature_variable]]
        model.fit(X_adjusted, y)
        y_adjusted_pred = model.predict(X_test)

        st.write(f"### Scenario Analysis for {target_variable} with {adjustment_factor}% Adjustment in {scenario_metric}")
        
        # Plot adjusted predictions vs actual
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_adjusted_pred, alpha=0.7)
        ax.plot(y_test, y_test, color='red')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Adjusted Predicted Values")
        ax.set_title(f"Adjusted {target_variable} Predictions vs Actual")
        st.pyplot(fig)

else:
    st.write("Please upload both stock price and income statement CSV files to continue.")
