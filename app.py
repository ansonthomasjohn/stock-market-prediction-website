import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objs as go

def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def prepare_data(data, window_size=5):
    data['Target'] = data['Close'].shift(-1) > data['Close']
    data.dropna(inplace=True)

    features = []
    targets = []

    for i in range(len(data) - window_size):
        features.append(data['Close'].values[i:i + window_size])
        targets.append(data['Target'].values[i + window_size])

    return pd.DataFrame(features), pd.Series(targets)

def plot_raw_data(data):
    fig = go.Figure()

    if 'Date' in data.columns and 'Open' in data.columns and 'Close' in data.columns:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.warning("Could not find required columns (Date, Open, Close) in the provided data.")

def main():
    st.title("Stock Market Prediction")

    symbol = st.text_input("Enter Stock Symbol", 'AAPL')
    start_date_str = st.text_input("Enter Start Date (YYYY-MM-DD)", '2023-01-01')
    end_date_str = st.text_input("Enter End Date (YYYY-MM-DD)", '2024-01-01')

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    stock_data = fetch_stock_data(symbol, start_date, end_date)
    features, targets = prepare_data(stock_data)

    if len(features) == 0:
        st.warning("No samples available for the given date range.")
        return

    test_size = min(0.2, len(features) - 1)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    predictions_list = [bool(pred) for pred in predictions]
    date_range = [start_date + timedelta(days=i) for i in range(len(predictions_list))]

    accuracy = accuracy_score(y_test, predictions)
    classification_report_output = classification_report(y_test, predictions)

    plot_raw_data(stock_data)

    st.subheader(f"Result for Stock Symbol: {symbol}")
    st.write(f"Model Accuracy: {accuracy}")
    st.text("Classification Report:")
    st.text(classification_report_output)
    st.text("Result Data:")
    st.text(list(zip(date_range, predictions_list)))

if __name__ == "__main__":
    main()
