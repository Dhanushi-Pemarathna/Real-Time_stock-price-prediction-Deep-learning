import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta



# Streamlit App Title
st.title('📈 Real-Time Stock Prediction App')

st.write("Original file is located at:")
st.markdown("(https://drive.google.com/drive/u/0/folders/1qBrzWoRiWWIucee_ah0460TYqDwk7I7b)")

# Load pre-trained models
@st.cache_resource
def load_models():
    classification_models = {
        'MLP': load_model('mlp_classification.h5'),
        'SimpleRNN': load_model('simplernn_classification.h5'),
        'LSTM': load_model('lstm_classification.h5')
    }
    regression_models = {
        'MLP': load_model('mlp_regression.h5'),
        'SimpleRNN': load_model('simplernn_regression.h5'),
        'LSTM': load_model('lstm_regression.h5')
    }
    return classification_models, regression_models

classification_models, regression_models = load_models()

# Fetch and preprocess stock data
@st.cache_data
def fetch_and_preprocess_data(ticker, days=60):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        return None, None, None

    df.fillna(method='ffill', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

    def create_sequences(data, sequence_length=10):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
        return np.array(sequences)

    sequence_data = create_sequences(df.values)
    return df, scaler, sequence_data

# Make predictions
def make_predictions(data, sequence_data, scaler):
    mlp_data = data.values[-10:].flatten().reshape(1, -1)

    # Classification
    cls_results = {}
    for name, model in classification_models.items():
        if 'lstm' in name.lower() or 'rnn' in name.lower():
            pred = model.predict(sequence_data[-1:])
        else:
            pred = model.predict(mlp_data)
        cls_results[name] = 'Up' if pred[0][0] > 0.5 else 'Down'

    # Regression
    reg_results = {}
    for name, model in regression_models.items():
        if 'lstm' in name.lower() or 'rnn' in name.lower():
            pred = model.predict(sequence_data[-1:])
        else:
            pred = model.predict(mlp_data)

        dummy = np.zeros((1, 5))
        dummy[0, 3] = pred[0][0]
        reg_results[name] = scaler.inverse_transform(dummy)[0, 3]

    return cls_results, reg_results

# Sidebar - User input
st.sidebar.header('📋 User Input Parameters')

ticker_list = ['AAPL', 'TSLA', 'GOOG', 'MSFT', 'AMZN', 'NFLX']
ticker = st.sidebar.selectbox('Select a Stock Ticker', ticker_list)

days_to_fetch = st.sidebar.slider('Days of historical data to fetch', 30, 365, 60)
model_type = st.sidebar.selectbox('Select model type for detailed view', ['MLP', 'SimpleRNN', 'LSTM'])
predict_clicked = st.sidebar.button('Predict')

# Main Logic
if predict_clicked:
    with st.spinner('Fetching data and making predictions...'):
        df, scaler, sequence_data = fetch_and_preprocess_data(ticker, days_to_fetch)

        if df is None:
            st.error("⚠️ No data found for this ticker. Please try another one.")
            st.stop()

        cls_results, reg_results = make_predictions(df, sequence_data, scaler)

        st.subheader(f'📊 Current Stock: {ticker}')

        # Chart: Closing Price Trend
        st.subheader('Stock Price Trend 📈')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, scaler.inverse_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])[:, 3], label='Close Price', color='cyan')
        ax.set_title(f'{ticker} Closing Price Trend', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Classification Results
        st.subheader('Price Movement Prediction (Up/Down) 🚀')
        cls_df = pd.DataFrame.from_dict(cls_results, orient='index', columns=['Prediction'])
        st.dataframe(cls_df.style.highlight_max(axis=0, color='lightgreen'))

        # Regression Results
        st.subheader('Next Day Closing Price Prediction 💲')
        reg_df = pd.DataFrame.from_dict(reg_results, orient='index', columns=['Predicted Price ($)'])
        st.dataframe(reg_df.style.highlight_min(axis=0, color='lightcoral'))

        # Detailed View
        st.subheader(f'Detailed Analysis: {model_type}')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Price Direction Prediction:** {cls_results[model_type]}")
            if cls_results[model_type] == 'Up':
                st.success('✅ The model predicts the price will increase tomorrow')
            else:
                st.error('⚠️ The model predicts the price will decrease tomorrow')

        with col2:
            dummy = np.zeros((1, 5))
            dummy[0, 3] = df['Close'].iloc[-1]
            current_price = scaler.inverse_transform(dummy)[0, 3]
            predicted_price = reg_results[model_type]
            change = predicted_price - current_price
            percent_change = (change / current_price) * 100

            st.metric(label="Current Close Price", value=f"${current_price:.2f}")
            st.metric(label="Predicted Close Price", value=f"${predicted_price:.2f}", delta=f"{percent_change:.2f}%")

        # Model Comparisons
        st.subheader('Model Performance Comparison 📊')

        # Updated Classification Accuracy
        st.markdown('**Classification Accuracy Comparison**')
        cls_acc = {
            'MLP': 0.54,
            'SimpleRNN': 0.44,
            'LSTM': 0.60
        }
        st.bar_chart(pd.DataFrame.from_dict(cls_acc, orient='index', columns=['Accuracy']))

        # Updated Regression RMSE
        st.markdown('**Regression RMSE Comparison**')
        reg_rmse = {
            'MLP': 8.3371,
            'SimpleRNN': 4.7176,
            'LSTM': 4.2724
        }
        st.bar_chart(pd.DataFrame.from_dict(reg_rmse, orient='index', columns=['RMSE']))

# Sidebar Information
st.sidebar.markdown("""---""")
st.sidebar.markdown("""
**App Instructions:**
1. Select a stock ticker (AAPL, TSLA, GOOG, etc.)
2. Choose how many days of historical data
3. Select the model you want to analyze in detail
4. Click **Predict** to fetch results!

**Models Used:**
- **MLP**: Multi-Layer Perceptron
- **SimpleRNN**: Simple Recurrent Neural Network
- **LSTM**: Long Short-Term Memory Network
""")
