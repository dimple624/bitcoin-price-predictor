Bitcoin Price Predictor

This project predicts the future price of Bitcoin using an LSTM (Long Short-Term Memory) neural network.
It has a backend built with Python and a frontend built with HTML, CSS, and JavaScript.

Features

Train an LSTM model on historical Bitcoin price data
Save the trained model as .h5 file
Predict future prices from recent data
Display predictions on a simple web interface

Installation

Clone the repository:

git clone https://github.com/dimple624/bitcoin-price-predictor.git
cd bitcoin-price-predictor/backend


Create and activate a virtual environment (optional but recommended):

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

Training the Model
python train_model.py


This will generate a model.h5 file inside the backend folder.

Running Predictions
python app.py

Frontend

Open frontend/index.html in your browser to view the prediction interface.

Requirements

tensorflow
numpy
pandas
scikit-learn
matplotlib
