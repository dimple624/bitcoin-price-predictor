import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load data
bitcoin_data = pd.read_csv('bitcoin_data.csv')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
bitcoin_data['Close_Normalized'] = scaler.fit_transform(np.array(bitcoin_data['Close']).reshape(-1, 1))

# Prepare data
def prepare_data(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10  # Example sequence length
X, y = prepare_data(bitcoin_data['Close_Normalized'].values, sequence_length)

# Reshape data for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
checkpoint = ModelCheckpoint('bitcoin_lstm_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

# Save model
model.save('bitcoin_lstm_model.h5')
print("Model saved successfully.")