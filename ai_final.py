import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(page_title="Crypto Predictor AI", layout="wide")

# Hiding TensorFlow warning logs to keep the console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_and_train_model(symbol):
    # Clearing the session to avoid cluttering memory when retraining
    K.clear_session()
    
    # Setting up a progress bar for the user interface
    progress_text = "Downloading data..."
    my_bar = st.progress(0, text=progress_text)

    try:
        # Downloading the data using yfinance
        df = yf.download(symbol, period='2y', interval='1d', progress=False)
        my_bar.progress(20, text="Data downloaded. Cleaning data...")
        
        # Check if data is empty
        if df.empty: return None, None, None, None

        # Handling MultiIndex columns if necessary (common issue with yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except: pass
        
        # Ensuring we have the required columns: Close price and Volume
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            # Fallback: trying to select columns by index if names don't match
            if len(df.columns) >= 5:
                 df = df.iloc[:, [3, 5]]
                 df.columns = ['Close', 'Volume']
            else:
                return None, None, None, None
        else:
            df = df[['Close', 'Volume']]

        # Filling missing values in Volume column
        df['Volume'] = df['Volume'].replace(0, np.nan).fillna(method='ffill')
        df = df.dropna()
        
        my_bar.progress(40, text="Training the AI model (LSTM)...")

        # SCALING DATA
        # We need to scale data between 0 and 1 for the LSTM model
        scaler_global = MinMaxScaler(feature_range=(0, 1))
        data_global = df[['Close', 'Volume']].values.astype(float)
        scaled_data = scaler_global.fit_transform(data_global)
        
        # We create a separate scaler for the target (Close price) to inverse transform later
        scaler_target = MinMaxScaler(feature_range=(0, 1))
        scaler_target.fit(df[['Close']].values.astype(float))
        
        x_train, y_train = [], []
        prediction_days = 60
        
        # Check if we have enough data points
        if len(scaled_data) < prediction_days + 10: return None, None, None, None

        # Creating the training sequences
        for i in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-prediction_days:i, :])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # BUILDING THE MODEL
        # Using a Sequential model with LSTM layers
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1)) # Output layer: prediction of the next price
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Training the model
        model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=0)
        
        my_bar.progress(100, text="Finished!")
        my_bar.empty() # Remove the progress bar
        
        return model, scaler_global, scaler_target, df

    except Exception as e:
        st.error(f"Technical error: {e}")
        return None, None, None, None

def get_prediction(ticker):
    # Train the model and get the scalers
    model, scaler_global, scaler_target, df = create_and_train_model(ticker)
    
    if model is None:
        st.error("Could not retrieve data for this ticker.")
        return None

    # MAKING THE PREDICTION
    # We take the last 60 days of data to predict the next day
    last_60_days = df[['Close', 'Volume']].values[-60:]
    scaled_last_60 = scaler_global.transform(last_60_days)
    
    # Reshaping for the model
    X_test = np.array([scaled_last_60])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

    # Predicting and inverse scaling to get the real price
    pred_price_scaled = model.predict(X_test, verbose=0)
    pred_price = scaler_target.inverse_transform(pred_price_scaled)
    
    current_price = float(df['Close'].iloc[-1])
    predicted_price = float(pred_price[0][0])
    
    return current_price, predicted_price, df

# ==========================================
# FRONTEND INTERFACE (STREAMLIT)
# ==========================================

# Title and Description
st.title("CryptoForecast AI")
st.markdown("""
This application uses an **LSTM (Long Short-Term Memory)** neural network to predict cryptocurrency prices based on historical **Price** and **Volume** data.
""")

# Sidebar for user input
with st.sidebar:
    st.header("Configuration")
    user_input = st.text_input("Crypto Symbol (e.g., BTC, ETH, SOL)", "BTC")
    st.info("The model is retrained in real-time for each request.")

# Action button
if st.button("Start Analysis", use_container_width=True):
    
    # Cleaning up the ticker input
    ticker = user_input.strip().upper()
    symbol_map = {'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD', 'SOLANA': 'SOL-USD'}
    if ticker in symbol_map: ticker = symbol_map[ticker]
    elif "-USD" not in ticker: ticker += "-USD"

    # Displaying a spinner while calculating
    with st.spinner(f"Analyzing {ticker}..."):
        result = get_prediction(ticker)

    if result:
        current, pred, df = result
        
        # Calculating percentage change
        delta = pred - current
        percent = (delta / current) * 100
        
        # Displaying metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Current Price", value=f"{current:.2f} $")
        
        with col2:
            st.metric(label="Prediction (Next Day)", value=f"{pred:.2f} $", delta=f"{percent:.2f}%")
            
        with col3:
            if percent > 0:
                st.success("Trend: UPWARD")
            else:
                st.error("Trend: DOWNWARD")

        # Interactive Chart
        st.subheader("History and Trend")
        
        # Plotting the closing price data
        chart_data = df['Close']
        st.line_chart(chart_data)
        
        # Technical details expander
        with st.expander("See technical details"):
            st.write("Latest data used (Price and Volume):")
            st.dataframe(df.tail())
