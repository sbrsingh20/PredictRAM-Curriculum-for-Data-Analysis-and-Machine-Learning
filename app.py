import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("Stock Price and Income Statement Analysis")
st.sidebar.header("Upload Data Files")

# File Uploads
stock_file = st.sidebar.file_uploader("Upload Stock Price Data (CSV)", type="csv")
income_file = st.sidebar.file_uploader("Upload Income Statement Data (CSV)", type="csv")

if stock_file and income_file:
    # Load the stock and income statement datasets
    stock_data = pd.read_csv(stock_file, parse_dates=['Date'])
    income_data = pd.read_csv(income_file, parse_dates=['Date'])

    # Convert dates in income data to datetime and resample stock data to quarterly frequency
    income_data['Date'] = pd.to_datetime(income_data['Date'], format='%b-%y')
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.set_index('Date').resample('Q').last().reset_index()

    # Display data previews
    st.write("### Stock Price Data Preview")
    st.dataframe(stock_data.head())
    st.write("### Income Statement Data Preview")
    st.dataframe(income_data.head())

    # Column selection
    st.sidebar.subheader("Select Columns for Analysis")
    stock_column = st.sidebar.selectbox("Choose Stock Data Column", options=stock_data.columns.drop('Date'))
    income_column = st.sidebar.selectbox("Choose Income Statement Column", options=income_data.columns.drop('Date'))

    # Merge datasets on Date
    merged_data = pd.merge(stock_data[['Date', stock_column]], income_data[['Date', income_column]], on='Date', how='inner')
    st.write("### Merged Data")
    st.dataframe(merged_data.head())

    # Module 1: Trend Analysis
    st.header("Trend Analysis")
    st.write(f"### Trend Chart for {stock_column} and {income_column}")
    fig, ax = plt.subplots()
    ax.plot(merged_data['Date'], merged_data[stock_column], label=stock_column, color='blue')
    ax.set_ylabel(stock_column, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    ax2.plot(merged_data['Date'], merged_data[income_column], label=income_column, color='green')
    ax2.set_ylabel(income_column, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    st.pyplot(fig)

    # Module 2: Correlation Analysis
    st.header("Correlation Analysis")
    correlation = merged_data[stock_column].corr(merged_data[income_column])
    st.write(f"Correlation between {stock_column} and {income_column}: {correlation:.4f}")

    # Module 3: Regression Analysis
    st.header("Regression Analysis")
    X = merged_data[[income_column]].values.reshape(-1, 1)
    y = merged_data[stock_column].values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Display regression metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-Squared: {r2:.4f}")

    # Plot actual vs predicted values
    st.write("### Actual vs Predicted Values")
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7, color='purple')
    ax.plot(y, y, color='red')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{stock_column} Prediction vs Actual")
    st.pyplot(fig)

else:
    st.write("Please upload both stock price and income statement CSV files to continue.")
