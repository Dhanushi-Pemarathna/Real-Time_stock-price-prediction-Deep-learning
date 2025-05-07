# Real-Time Stock Price Prediction - Deep Learning

A Streamlit-powered web application for **real-time stock price prediction** using **machine learning**. Features include live data fetching, historical stock visualization, and future price forecasting based on trained models.

## Overview

This repository contains a real-time stock price prediction application powered by **Streamlit** and various **Deep Learning models** such as **LSTM**, **MLP**, and **SimpleRNN**. The app allows users to input a stock ticker symbol, fetch live stock data, visualize historical stock prices, and make future price predictions based on trained models. The deep learning models are pre-trained and saved in `.h5` files, and users can use the app to predict stock prices with just a few clicks.

### Key Features:
- **Live Data Fetching**: Fetches real-time stock data from external APIs.
- **Historical Stock Visualization**: Visualizes stock prices over time.
- **Stock Price Forecasting**: Makes predictions on future stock prices based on trained deep learning models.

## Repository Structure

The repository is organized as follows:

- deep_stock.ipynb # Jupyter notebook for data preprocessing, model training, and evaluation
- deep_stock_UI.py # Streamlit app for real-time stock prediction user interface
- .h5 files # Pre-trained model files (LSTM, MLP, SimpleRNN)
- scaler.save # Pre-saved scaler for feature normalization
- requirements.txt # Python dependencies for the project


### File Descriptions:

- **`deep_stock.ipynb`**: Jupyter notebook containing the steps for data collection, cleaning, preprocessing, model training, and evaluation. This notebook allows you to explore and train the models yourself.
  
- **`deep_stock_UI.py`**: Python script that implements the user interface using **Streamlit**. It allows you to interact with the stock prediction models in real-time by inputting stock ticker symbols.

- **`.h5 files`**: These are the pre-trained deep learning model files saved in **HDF5 format**. They include:
    - `lstm_classification.h5`: Pre-trained LSTM model for stock classification.
    - `lstm_regression.h5`: Pre-trained LSTM model for stock price regression.
    - `mlp_classification.h5`: Pre-trained MLP model for stock classification.
    - `mlp_regression.h5`: Pre-trained MLP model for stock price regression.
    - `simplernn_classification.h5`: Pre-trained SimpleRNN model for stock classification.
    - `simplernn_regression.h5`: Pre-trained SimpleRNN model for stock price regression.

- **`scaler.save`**: A saved **scaler model** used to normalize the input features before feeding them to the neural network models.

## Installation

Follow these steps to set up the environment and run the project on your local machine:

### 1. Clone the Repository
Clone the repository to your local machine:

bash
git clone https://github.com/Dhanushi-Pemarathna/real-time-stock-predictor---Deep-Learning.git

## Usage
1. Running the Jupyter Notebook
To run the Jupyter Notebook (deep_stock.ipynb), which contains the full workflow for data preprocessing, model training, and evaluation. After running this file on jupiter notebook or colab it will download some .h5 file. after you have to run streamlit UI.

2. Running the Streamlit UI
To run the Streamlit app (deep_stock_UI.py), which allows real-time stock prediction:
streamlit run deep_stock_UI.py
first go to the command prompt this file's location. This will launch a local web server.

## Models and Files
Pre-trained Models:
LSTM models (lstm_classification.h5, lstm_regression.h5): These models use Long Short-Term Memory networks for time series forecasting.

MLP models (mlp_classification.h5, mlp_regression.h5): These models use Multilayer Perceptrons for predicting stock prices.

SimpleRNN models (simplernn_classification.h5, simplernn_regression.h5): These models use Simple Recurrent Networks for stock prediction.

## Scaler:
The scaler.save file is used for normalizing the input data before passing it to the model. This ensures that the features are on the same scale, improving model performance.

## Dependencies
The following Python libraries are required to run the project:

TensorFlow
Keras
Streamlit
pandas
numpy
matplotlib
scikit-learn

You can install them by running:
pip install tensorflow keras streamlit pandas numpy matplotlib scikit-learn

