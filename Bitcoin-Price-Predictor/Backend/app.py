from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import requests

app = Flask(__name__)

# Load data
bitcoin_data = pd.read_csv('bitcoin_data.csv')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
bitcoin_data['Close_Normalized'] = scaler.fit_transform(np.array(bitcoin_data['Close']).reshape(-1, 1))

# Load LSTM model
model = load_model('bitcoin_lstm_model.h5')

@app.route('/')
def index():
    # Fetch current price from CoinGecko API
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
    #current_price = response.json()['bitcoin']['usd']
    return render_template('index.html')#, current_price=current_price)

@app.route('/predict', methods=['POST'])
def predict():
    # Get number of future days to predict
    future_days = int(request.form['future_days'])

    # Prepare input for prediction
    last_sequence = bitcoin_data['Close_Normalized'].values[-10:]  # Assuming sequence length of 10
    last_sequence = np.array(last_sequence).reshape(1, -1)

    # Generate predictions
    predictions = []
    for _ in range(future_days):
        prediction = model.predict(last_sequence.reshape(1, 10, 1))
        predictions.append(prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1] = prediction

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Prepare data to send to frontend
    prediction_dates = pd.date_range(start=bitcoin_data['Date'].iloc[-1], periods=future_days + 1)[1:]
    prediction_dates_str = [date.strftime('%Y-%m-%d') for date in prediction_dates]
    prediction_values = predictions.tolist()

    #current_price = bitcoin_data['Close'].iloc[-1]
    return render_template('index.html', dates=prediction_dates_str, prices=prediction_values)#, current_price=current_price)

@app.context_processor
def utility_processor():
    def zip_lists(a, b):
        return zip(a, b)
    return dict(zip=zip_lists)

if __name__ == '__main__':
    app.run(debug=True)