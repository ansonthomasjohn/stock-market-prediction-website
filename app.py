from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data['Close'].to_frame()

def prepare_data(data, window_size=5):
    data['Target'] = data['Close'].shift(-1) > data['Close']
    data.dropna(inplace=True)
    
    features = []
    targets = []

    for i in range(len(data) - window_size):
        features.append(data['Close'].values[i:i+window_size])
        targets.append(data['Target'].values[i+window_size])

    return pd.DataFrame(features), pd.Series(targets)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form.get('symbol', 'AAPL')
    start_date_str = request.form.get('start_date', '2023-01-01')
    end_date_str = request.form.get('end_date', '2024-01-01')

    # Convert start_date and end_date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    stock_data = fetch_stock_data(symbol, start_date, end_date)
    features, targets = prepare_data(stock_data)

    if len(features) == 0:
        # Handle the case where there are no samples
        return render_template('result.html', stock_symbol=symbol, accuracy='N/A', classification_report='N/A', result_data=[])

    # Adjust test_size based on the available data
    test_size = min(0.2, len(features) - 1)  # Ensure test_size is not larger than the number of samples - 1

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    # Convert NumPy boolean array to Python list
    predictions_list = [bool(pred) for pred in predictions]

    # Generate dates from start_date to end_date
    date_range = [start_date + timedelta(days=i) for i in range(len(predictions_list))]

    accuracy = accuracy_score(y_test, predictions)
    classification_report_output = classification_report(y_test, predictions)
    result_data = list(zip(date_range, predictions_list))
    return render_template('result.html', stock_symbol=symbol, accuracy=accuracy, classification_report=classification_report_output, result_data=result_data)

if __name__ == "__main__":
    app.run(debug=True)
