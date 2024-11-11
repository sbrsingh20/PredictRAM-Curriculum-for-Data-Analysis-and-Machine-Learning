import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Stock Price and Income Statement Analysis")

# File upload
st.sidebar.header("Upload Files")
income_file = st.sidebar.file_uploader("Upload Income Statement CSV", type="csv")
stock_file = st.sidebar.file_uploader("Upload Stock Price Data CSV", type="csv")

if income_file and stock_file:
    # Load the files
    income_df = pd.read_csv(income_file)
    stock_df = pd.read_csv(stock_file)

    # Data overview
    st.write("### Income Statement Data")
    st.write(income_df.head())

    st.write("### Stock Price Data")
    st.write(stock_df.head())

    # Convert date columns to datetime format
    income_df['Date'] = pd.to_datetime(income_df['Date'], format='%b-%y')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')

    # User selects columns for analysis
    income_column = st.sidebar.selectbox("Select Income Statement Column", income_df.columns[1:])
    stock_column = st.sidebar.selectbox("Select Stock Data Column", stock_df.columns[1:])

    # Resampling stock data to quarterly frequency to match income statement
    stock_df.set_index('Date', inplace=True)
    stock_df = stock_df.resample('Q').mean()
    income_df.set_index('Date', inplace=True)

    # Merging data on date to align for analysis
    merged_data = pd.merge(income_df[[income_column]], stock_df[[stock_column]], left_index=True, right_index=True, how='inner')

    st.write("### Merged Data for Analysis")
    st.write(merged_data)

    # Check and handle NaN values
    if merged_data.isnull().any().any():
        st.warning("Merged data contains NaN values. Dropping NaNs for analysis.")
        merged_data = merged_data.dropna()

    # Ensure there is data after merging and NaN removal
    if merged_data.empty:
        st.error("No data available for analysis after merging and NaN handling. Please check the date range and data quality in your files.")
    else:
        # Display correlation analysis
        correlation = merged_data.corr().iloc[0, 1]
        st.write(f"### Correlation between {income_column} and {stock_column}: {correlation:.2f}")

        # Plotting trend chart
        st.write("### Trend Analysis")
        plt.figure(figsize=(10, 5))
        plt.plot(merged_data.index, merged_data[income_column], label=income_column, color='blue')
        plt.plot(merged_data.index, merged_data[stock_column], label=stock_column, color='red')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.title("Trend Analysis of Selected Columns")
        st.pyplot(plt)

        # Regression Analysis
        st.write("### Regression Analysis")
        
        # Prepare X and y for regression
        X = merged_data[[income_column]].values.reshape(-1, 1)  # Ensure X is a 2D array
        y = merged_data[stock_column].values                    # Ensure y is a 1D array

        model = LinearRegression()
        try:
            model.fit(X, y)
            y_pred = model.predict(X)

            # Display regression results
            st.write(f"Regression Coefficient (Slope): {model.coef_[0]:.2f}")
            st.write(f"Regression Intercept: {model.intercept_:.2f}")

            # Plotting regression results
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=merged_data[income_column], y=merged_data[stock_column], color='blue', label="Actual Data")
            plt.plot(merged_data[income_column], y_pred, color='red', label="Regression Line")
            plt.xlabel(income_column)
            plt.ylabel(stock_column)
            plt.title(f"Regression Analysis of {income_column} vs {stock_column}")
            plt.legend()
            st.pyplot(plt)
            
        except ValueError as e:
            st.error(f"Regression model fitting failed: {e}")

else:
    st.warning("Please upload both the income statement and stock price data files.")
