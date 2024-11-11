import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# File upload for income statement and stock data
st.sidebar.header("Upload Data Files")
income_file = st.sidebar.file_uploader("Upload Income Statement CSV", type="csv")
stock_file = st.sidebar.file_uploader("Upload Stock Data CSV", type="csv")

if income_file is not None and stock_file is not None:
    # Load data
    income_df = pd.read_csv(income_file)
    stock_df = pd.read_csv(stock_file)

    # Parse date columns and set them as index
    income_df['Date'] = pd.to_datetime(income_df['Date'], format='%b-%y', errors='coerce')
    income_df.set_index('Date', inplace=True)

    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
    stock_df.set_index('Date', inplace=True)

    # Sidebar - Column selection and Date Range selection
    st.sidebar.header("Options")
    income_column = st.sidebar.selectbox("Select Income Statement Column", income_df.columns)
    stock_column = st.sidebar.selectbox("Select Stock Data Column", stock_df.columns)

    # Display date range of data available
    st.write("Income Data Date Range:", income_df.index.min(), "to", income_df.index.max())
    st.write("Stock Data Date Range:", stock_df.index.min(), "to", stock_df.index.max())

    # Select start and end dates
    start_date = st.sidebar.date_input("Start Date", value=income_df.index.min())
    end_date = st.sidebar.date_input("End Date", value=income_df.index.max())

    # Filter data based on selected date range
    income_df_filtered = income_df.loc[start_date:end_date, [income_column]]
    stock_df_filtered = stock_df.loc[start_date:end_date, [stock_column]]

    # Display filtered data before merging
    st.write("Filtered Income Data", income_df_filtered.head())
    st.write("Filtered Stock Data", stock_df_filtered.head())

    # Merge filtered data
    merged_data = pd.merge(income_df_filtered, stock_df_filtered, left_index=True, right_index=True, how='inner')
    merged_data.columns = [income_column, stock_column]

    # Drop NaN values
    merged_data = merged_data.dropna()

    if merged_data.empty:
        st.error("No data available for analysis after merging and NaN handling. Please check the date range and data quality in your files.")
    else:
        st.write("Merged Data", merged_data)

        # Plot trend for each selected column
        st.write(f"Trend for {income_column} and {stock_column}")
        fig, ax = plt.subplots()
        ax.plot(merged_data.index, merged_data[income_column], label=income_column, color='blue')
        ax.set_ylabel(income_column, color='blue')
        ax2 = ax.twinx()
        ax2.plot(merged_data.index, merged_data[stock_column], label=stock_column, color='red')
        ax2.set_ylabel(stock_column, color='red')
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        st.pyplot(fig)

        # Regression Analysis
        st.write("### Regression Analysis")
        X = merged_data[[income_column]].values
        y = merged_data[stock_column].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Display regression results
        st.write(f"Intercept: {model.intercept_}")
        st.write(f"Coefficient: {model.coef_[0]}")
        st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred)}")
        st.write(f"R^2 Score: {r2_score(y, y_pred)}")

        # Plot regression
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label="Actual Data")
        ax.plot(X, y_pred, color='red', label="Regression Line")
        ax.set_xlabel(income_column)
        ax.set_ylabel(stock_column)
        ax.legend()
        st.pyplot(fig)
else:
    st.write("Please upload both Income Statement and Stock Data CSV files to proceed.")
