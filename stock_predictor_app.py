# Stock Prediction System

import os
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

STOCKS_DIR = 'Stocks'

# Helper to list available stocks
def list_stocks():
    return [f[:-4] for f in os.listdir(STOCKS_DIR) if f.endswith('.csv')]

# Load stock data
def load_stock(symbol):
    df = pd.read_csv(os.path.join(STOCKS_DIR, f'{symbol}.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

# Resample data
def resample_data(df, freq):
    df = df.set_index('date')
    if freq == 'Monthly':
        df = df.resample('M').last()
    else:
        df = df.resample('D').last().dropna()
    return df

# Split train/test (last 3 months for test)
def train_test_split_by_date(df, freq):
    last_date = df.index[-1]
    if freq == 'Monthly':
        test_start = last_date - pd.DateOffset(months=3)
    else:
        test_start = last_date - pd.DateOffset(months=3)
    train = df[df.index < test_start]['close'].dropna().values
    test = df[df.index >= test_start]['close'].dropna().values
    test_dates = df[df.index >= test_start].index
    return train, test, test_dates

def create_features(df, freq):
    df_feat = df.copy()
    df_feat['lag1'] = df_feat['close'].shift(1)
    df_feat['lag2'] = df_feat['close'].shift(2)
    df_feat['lag3'] = df_feat['close'].shift(3)
    df_feat['rolling_mean_5'] = df_feat['close'].rolling(window=5).mean()
    df_feat['rolling_std_5'] = df_feat['close'].rolling(window=5).std()
    df_feat['ema_5'] = df_feat['close'].ewm(span=5, adjust=False).mean()
    df_feat['ema_10'] = df_feat['close'].ewm(span=10, adjust=False).mean()
    df_feat['returns'] = df_feat['close'].pct_change()
    df_feat['lagged_returns'] = df_feat['returns'].shift(1)
    if freq == 'Daily':
        df_feat['dayofweek'] = df_feat.index.dayofweek
        df_feat['day'] = df_feat.index.day
        df_feat['month'] = df_feat.index.month
    else:
        df_feat['month'] = df_feat.index.month
    df_feat = df_feat.dropna()
    return df_feat

# ARIMA Model
def arima_forecast(series, steps):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# SARIMA Model
def sarima_forecast(series, steps):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast

# XGBoost Model
def xgb_forecast_with_features(df_feat, train_idx, test_idx):
    features = ['lag1', 'lag2', 'lag3', 'rolling_mean_5', 'rolling_std_5', 'month']
    if 'dayofweek' in df_feat.columns:
        features += ['dayofweek', 'day']
    X_train = df_feat.iloc[train_idx][features]
    y_train = df_feat.iloc[train_idx]['close']
    X_test = df_feat.iloc[test_idx][features]
    model = XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds

# LSTM Model
def lstm_forecast_with_features(df_feat, train_idx, test_idx):
    features = ['lag1', 'lag2', 'lag3', 'rolling_mean_5', 'rolling_std_5', 'month']
    if 'dayofweek' in df_feat.columns:
        features += ['dayofweek', 'day']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_feat[features])
    y = df_feat['close'].values
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=20, verbose=0)
    preds = model.predict(X_test_lstm).flatten()
    return preds

# Streamlit UI
st.title('Stock Prediction System')
st.write('Select a stock and prediction frequency:')
stocks = list_stocks()
stock_options = stocks + ['Combined']
stock = st.selectbox('Stock', stock_options)
freq = st.radio('Prediction Frequency', ['Daily', 'Monthly'])
steps = st.slider('Prediction Steps', 1, 30, 7)

if st.button('Predict'):

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if stock != 'Combined':
        df = load_stock(stock)
        df = resample_data(df, freq)
        series = df['close'].dropna().values
        st.write(f'Last available date: {df.index[-1].date()}')
        st.line_chart(df['close'])

        # Train/test split
        train, test, test_dates = train_test_split_by_date(df, freq)
        st.write(f"Training until {test_dates[0].date()}, testing from {test_dates[0].date()} to {test_dates[-1].date()} ({len(test)} samples)")

        # Feature engineering for ML models
        df_feat = create_features(df, freq)
        min_date = df_feat.index.min()
        max_date = pd.Timestamp.today()

        train_mask = df_feat.index < test_dates[0]
        test_mask = df_feat.index >= test_dates[0]
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Predict for the test set for evaluation metrics and charts
        xgb_pred = xgb_forecast_with_features(df_feat, train_idx, test_idx)
        lstm_pred = lstm_forecast_with_features(df_feat, train_idx, test_idx)
        # Define test_feat for evaluation
        test_feat = df_feat.iloc[test_idx]['close'].values

        # Latest 5 days prediction tables with high, low, open, volume
        latest_df = df.iloc[-5:]
        latest_dates = latest_df.index.strftime('%Y-%m-%d')
        actual_close = np.round(latest_df['close'].values, 2)
        actual_high = np.round(latest_df['high'].values, 2) if 'high' in latest_df.columns else [None]*5
        actual_low = np.round(latest_df['low'].values, 2) if 'low' in latest_df.columns else [None]*5
        actual_open = np.round(latest_df['open'].values, 2) if 'open' in latest_df.columns else [None]*5

        # Get corresponding predictions for last 5 test days
        xgb_latest = np.round(xgb_pred[-5:], 2)
        lstm_latest = np.round(lstm_pred[-5:], 2)

        # Table for XGBoost
        xgb_table = pd.DataFrame({
            'Date': latest_dates,
            'Actual Close': actual_close,
            'High': actual_high,
            'Low': actual_low,
            'Open': actual_open,
            'XGBoost Predicted': xgb_latest,
            'XGBoost Diff': np.round(actual_close - xgb_latest, 2)
        })
        st.write('Latest 5 Days XGBoost Prediction Table:')
        st.table(xgb_table)

        # Table for LSTM
        lstm_table = pd.DataFrame({
            'Date': latest_dates,
            'Actual Close': actual_close,
            'High': actual_high,
            'Low': actual_low,
            'Open': actual_open,
            'LSTM Predicted': lstm_latest,
            'LSTM Diff': np.round(actual_close - lstm_latest, 2)
        })
        st.write('Latest 5 Days LSTM Prediction Table:')
        st.table(lstm_table)

        st.write('XGBoost (with features) Evaluation:')
        mape_xgb = f"{round(mape(test_feat, xgb_pred), 2)}%"
        st.write({
            'MAE': round(mean_absolute_error(test_feat, xgb_pred), 3),
            'RMSE': round(np.sqrt(mean_squared_error(test_feat, xgb_pred)), 3),
            'MAPE': mape_xgb
        })
        # Plot train/test actual and test predicted for XGBoost
        train_actual = np.round(df_feat.iloc[train_idx]['close'].values, 2)
        test_actual = np.round(df_feat.iloc[test_idx]['close'].values, 2)
        test_pred = np.round(xgb_pred, 2)
        plot_df_xgb = pd.DataFrame({
            'Train Actual': np.concatenate([train_actual, np.full(len(test_actual), np.nan)]),
            'Test Actual': np.concatenate([np.full(len(train_actual), np.nan), test_actual]),
            'Test Predicted': np.concatenate([np.full(len(train_actual), np.nan), test_pred])
        }, index=np.concatenate([df_feat.iloc[train_idx].index, df_feat.iloc[test_idx].index]))
        st.write('XGBoost Prediction Graph (Train/Test split)')
        st.line_chart(plot_df_xgb, y=['Train Actual', 'Test Actual', 'Test Predicted'])

        st.write('LSTM (with features) Evaluation:')
        mape_lstm = f"{round(mape(test_feat, lstm_pred), 2)}%"
        st.write({
            'MAE': round(mean_absolute_error(test_feat, lstm_pred), 3),
            'RMSE': round(np.sqrt(mean_squared_error(test_feat, lstm_pred)), 3),
            'MAPE': mape_lstm
        })
        # Plot train/test actual and test predicted for LSTM
        test_pred_lstm = np.round(lstm_pred, 2)
        plot_df_lstm = pd.DataFrame({
            'Train Actual': np.concatenate([train_actual, np.full(len(test_actual), np.nan)]),
            'Test Actual': np.concatenate([np.full(len(train_actual), np.nan), test_actual]),
            'Test Predicted': np.concatenate([np.full(len(train_actual), np.nan), test_pred_lstm])
        }, index=np.concatenate([df_feat.iloc[train_idx].index, df_feat.iloc[test_idx].index]))
        st.write('LSTM Prediction Graph (Train/Test split)')
        st.line_chart(plot_df_lstm, y=['Train Actual', 'Test Actual', 'Test Predicted'])
    else:
        # Combined mode: aggregate predictions for all stocks, show only table
        combined_rows = []
        for symbol in stocks:
            df = load_stock(symbol)
            df = resample_data(df, freq)
            # Feature engineering
            df_feat = create_features(df, freq)
            if len(df_feat) < 10:
                continue  # skip if not enough data
            # Train/test split
            last_date = df.index[-1]
            if freq == 'Monthly':
                test_start = last_date - pd.DateOffset(months=3)
            else:
                test_start = last_date - pd.DateOffset(months=3)
            train_mask = df_feat.index < test_start
            test_mask = df_feat.index >= test_start
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(test_idx) < 1:
                continue  # skip if not enough test data
            # Predict for the test set
            xgb_pred = xgb_forecast_with_features(df_feat, train_idx, test_idx)
            test_feat = df_feat.iloc[test_idx]['close'].values
            # Only show the latest date
            latest_idx = -1
            latest_date = df.index[latest_idx].strftime('%Y-%m-%d')
            actual_close = np.round(df.iloc[latest_idx]['close'], 2)
            actual_high = np.round(df.iloc[latest_idx]['high'], 2) if 'high' in df.columns else None
            actual_low = np.round(df.iloc[latest_idx]['low'], 2) if 'low' in df.columns else None
            predicted = np.round(xgb_pred[latest_idx], 2) if len(xgb_pred) > abs(latest_idx) else None
            rmse = round(np.sqrt(mean_squared_error(test_feat, xgb_pred)), 3)
            mae = round(mean_absolute_error(test_feat, xgb_pred), 3)
            mape_val = f"{round(mape(test_feat, xgb_pred), 2)}%"
            combined_rows.append({
                'Stock': symbol,
                'Date': latest_date,
                'Actual': actual_close,
                'High': actual_high,
                'Low': actual_low,
                'Predicted': predicted,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape_val
            })
        if combined_rows:
            combined_table = pd.DataFrame(combined_rows)
            # Format only numeric columns, leave missing as np.nan
            for col in ['Actual', 'High', 'Low', 'Predicted', 'RMSE', 'MAE']:
                combined_table[col] = pd.to_numeric(combined_table[col], errors='coerce')
            # Round numeric columns
            combined_table['Actual'] = combined_table['Actual'].round(2)
            combined_table['High'] = combined_table['High'].round(2)
            combined_table['Low'] = combined_table['Low'].round(2)
            combined_table['Predicted'] = combined_table['Predicted'].round(2)
            combined_table['RMSE'] = combined_table['RMSE'].round(3)
            combined_table['MAE'] = combined_table['MAE'].round(3)
            # Show table with st.dataframe (no Styler formatting)
            st.write('Combined Stocks Prediction Table (XGBoost, last 5 days for each stock):')
            st.dataframe(combined_table, use_container_width=True)
        else:
            st.write('Not enough data to show combined table.')