# =============================================================================
# ARCHIVO: app/utils.py
# Utilidades para el API
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100.0

def save_model_artifacts(model, scaler_x, scaler_y, lag_size, metrics=None):
    """Guardar modelo y scalers"""
    joblib.dump(model, f'app/models/best_model_lag{lag_size}.pkl')
    joblib.dump(scaler_x, f'app/models/scaler_x_lag{lag_size}.pkl')
    joblib.dump(scaler_y, f'app/models/scaler_y_lag{lag_size}.pkl')
    
    if metrics:
        pd.DataFrame([metrics]).to_csv(f'app/models/metrics_lag{lag_size}.csv', index=False)

def load_btc_data(filepath='btc_1d_data_2018_to_2025.csv'):
    """Cargar y preparar datos de Bitcoin"""
    btc = pd.read_csv(filepath)
    btc = btc[["Open time", "Close"]].dropna().reset_index(drop=True)
    btc = btc.rename(columns={"Open time": "Date"})
    btc["Close"] = btc["Close"].astype(float)
    btc["LogReturn"] = np.log(btc["Close"] / btc["Close"].shift(1))
    
    window_size = 7
    btc["Volatility"] = btc["LogReturn"].rolling(window=window_size).std() * np.sqrt(365)
    btc = btc.sort_values("Date").reset_index(drop=True).dropna().reset_index(drop=True)
    
    return btc
