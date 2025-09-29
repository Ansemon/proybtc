import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tsxv.splitTrainValTest import split_train_val_test_groupKFold

# Crear directorio para modelos
os.makedirs('app/models', exist_ok=True)

# Cargar datos (igual que en tu entrenamiento)
btc = pd.read_csv("btc_1d_data_2018_to_2025.csv")
btc = btc[["Open time", "Close"]].dropna().reset_index(drop=True)
btc = btc.rename(columns={"Open time": "Date"})
btc["Close"] = btc["Close"].astype(float)
btc["LogReturn"] = np.log(btc["Close"] / btc["Close"].shift(1))

window_size = 7
btc["Volatility"] = btc["LogReturn"].rolling(window=window_size).std() * np.sqrt(365)
btc = btc.sort_values("Date").reset_index(drop=True).dropna().reset_index(drop=True)

volatility_full = btc["Volatility"].values

# Configuraciones ganadoras según tus resultados
best_configs = {
    7: {
        'hidden_layer_sizes': (200, 100),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'scaler': 'standard'
    },
    14: {
        'hidden_layer_sizes': (30,),
        'activation': 'relu',
        'alpha': 0.01,
        'learning_rate_init': 0.005,
        'max_iter': 500,
        'scaler': 'standard'
    },
    21: {
        'hidden_layer_sizes': (10,),
        'activation': 'relu',
        'alpha': 1e-5,
        'learning_rate_init': 0.005,
        'max_iter': 500,
        'scaler': 'standard'
    },
    28: {
        'hidden_layer_sizes': (30, 15),
        'activation': 'tanh',
        'alpha': 0.001,
        'learning_rate_init': 0.0005,
        'max_iter': 1000,
        'scaler': 'standard'
    }
}

# Métricas obtenidas de tus resultados
metrics_results = {
    7: {'MAE_mean': 0.155308, 'RMSE_mean': 0.207478, 'MAPE_mean': 34.169668, 'BDS_mean': 0.614925},
    14: {'MAE_mean': 0.184337, 'RMSE_mean': 0.331413, 'MAPE_mean': 35.871253, 'BDS_mean': 0.534328},
    21: {'MAE_mean': 0.163318, 'RMSE_mean': 0.204845, 'MAPE_mean': 42.112820, 'BDS_mean': 0.698507},
    28: {'MAE_mean': 0.181783, 'RMSE_mean': 0.232718, 'MAPE_mean': 40.411510, 'BDS_mean': 0.602985}
}

n_steps_out = 7
n_steps_jump = 1

print("Entrenando y guardando modelos finales...")
print("=" * 50)

for lag_size in [7, 14, 21, 28]:
    print(f"\nProcesando LAG {lag_size}...")
    
    # Obtener datos
    X, y, Xcv, ycv, Xtest, ytest = split_train_val_test_groupKFold(
        volatility_full,
        lag_size,
        n_steps_out,
        n_steps_jump
    )
    
    # Configuración del mejor modelo para este lag
    cfg = best_configs[lag_size]
    
    # Combinar TODOS los datos disponibles para entrenar el modelo final
    # (esto asume que quieres el mejor modelo posible para deployment)
    all_X = []
    all_y = []
    
    # Concatenar train, val y test de todos los folds
    for fold in range(len(X)):
        all_X.append(X[fold])
        all_X.append(Xcv[fold])
        all_X.append(Xtest[fold])
        all_y.append(y[fold])
        all_y.append(ycv[fold])
        all_y.append(ytest[fold])
    
    # Combinar todo
    X_final = np.vstack(all_X)
    y_final = np.vstack(all_y)
    
    print(f"  Datos finales: {X_final.shape[0]} muestras")
    
    # Scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Escalar
    X_scaled = scaler_x.fit_transform(X_final)
    y_scaled = scaler_y.fit_transform(y_final)
    
    # Crear modelo final
    final_model = MLPRegressor(
        hidden_layer_sizes=cfg['hidden_layer_sizes'],
        activation=cfg['activation'],
        alpha=cfg['alpha'],
        learning_rate_init=cfg['learning_rate_init'],
        max_iter=cfg['max_iter'],
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15
    )
    
    # Entrenar modelo final
    print(f"  Entrenando modelo con config: {cfg}")
    final_model.fit(X_scaled, y_scaled)
    print(f"  Entrenamiento completado. Iteraciones: {final_model.n_iter_}")
    
    # Guardar modelo y scalers
    joblib.dump(final_model, f'app/models/best_model_lag{lag_size}.pkl')
    joblib.dump(scaler_x, f'app/models/scaler_x_lag{lag_size}.pkl')
    joblib.dump(scaler_y, f'app/models/scaler_y_lag{lag_size}.pkl')
    
    # Guardar métricas
    metrics = {
        'lag_size': lag_size,
        'model_type': 'MLP',
        'architecture': str(cfg['hidden_layer_sizes']),
        'activation': cfg['activation'],
        'alpha': cfg['alpha'],
        'learning_rate_init': cfg['learning_rate_init'],
        'max_iter': cfg['max_iter'],
        **metrics_results[lag_size]
    }
    
    pd.DataFrame([metrics]).to_csv(f'app/models/metrics_lag{lag_size}.csv', index=False)
    
    print(f"  ✓ Modelo guardado: app/models/best_model_lag{lag_size}.pkl")
    print(f"  ✓ Scalers guardados")
    print(f"  ✓ Métricas: RMSE={metrics['RMSE_mean']:.4f}, MAPE={metrics['MAPE_mean']:.2f}%")

print("\n" + "=" * 50)
print("TODOS LOS MODELOS GUARDADOS EXITOSAMENTE")
print("=" * 50)

# Verificar archivos creados
print("\nArchivos creados:")
for lag_size in [7, 14, 21, 28]:
    files = [
        f'app/models/best_model_lag{lag_size}.pkl',
        f'app/models/scaler_x_lag{lag_size}.pkl', 
        f'app/models/scaler_y_lag{lag_size}.pkl',
        f'app/models/metrics_lag{lag_size}.csv'
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file_path} (NO ENCONTRADO)")

print(f"\nYa puedes ejecutar la API:")
print("uvicorn app.api:app --reload")
print("Luego visita: http://localhost:8000/docs")

# Test rápido de un modelo
print(f"\nTest rápido del modelo LAG 7:")
try:
    model = joblib.load('app/models/best_model_lag7.pkl')
    scaler_x = joblib.load('app/models/scaler_x_lag7.pkl') 
    scaler_y = joblib.load('app/models/scaler_y_lag7.pkl')
    
    # Datos de prueba
    test_input = np.array([[0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5]])
    test_scaled = scaler_x.transform(test_input)
    pred_scaled = model.predict(test_scaled)
    prediction = scaler_y.inverse_transform(pred_scaled)
    
    print(f"  Input: {test_input[0]}")
    print(f"  Predicción (7 días): {prediction[0]}")
    print("  ✓ Test exitoso!")
    
except Exception as e:
    print(f"  ✗ Error en test: {e}")