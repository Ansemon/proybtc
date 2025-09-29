
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

app = FastAPI(
    title="Bitcoin Volatility Prediction API",
    description="API para predecir volatilidad de Bitcoin usando lags de precios históricos",
    version="1.0.0"
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos y scalers cargados globalmente
models = {}
scalers = {}

def load_models():
    """Cargar modelos entrenados y scalers"""
    try:
        lag_sizes = [7, 14, 21, 28]
        for lag in lag_sizes:
            models[lag] = joblib.load(f'app/models/best_model_lag{lag}.pkl')
            scalers[lag] = {
                'scaler_x': joblib.load(f'app/models/scaler_x_lag{lag}.pkl'),
                'scaler_y': joblib.load(f'app/models/scaler_y_lag{lag}.pkl')
            }
        logger.info("Modelos cargados exitosamente")
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        raise e

# Cargar modelos al iniciar
@app.on_event("startup")
async def startup_event():
    load_models()

# Modelos Pydantic para request/response
class PredictionRequest(BaseModel):
    lags: List[float]
    
    @validator('lags')
    def validate_lags(cls, v):
        if len(v) not in [7, 14, 21, 28]:
            raise ValueError('lags debe tener longitud 7, 14, 21 o 28')
        if any(x <= 0 for x in v):
            raise ValueError('Todos los valores de lags deben ser positivos')
        return v

class PredictionResponse(BaseModel):
    volatility_forecast: List[float]
    horizon_days: int
    lag_size: int
    model_info: dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[int]

# Endpoints
@app.get("/", response_model=dict)
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Bitcoin Volatility Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predecir volatilidad",
            "/health": "GET - Estado de la API",
            "/docs": "GET - Documentación automática"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_volatility(request: PredictionRequest):
    """
    Predecir volatilidad de Bitcoin para los próximos 7 días
    
    Args:
        request: Contiene lags de volatilidad histórica
        
    Returns:
        Predicción de volatilidad para 7 días
    """
    try:
        lags = np.array(request.lags)
        lag_size = len(lags)
        
        # Verificar que tenemos el modelo para este lag size
        if lag_size not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Modelo no disponible para lag_size={lag_size}"
            )
        
        # Preparar datos
        X = lags.reshape(1, -1)
        
        # Escalar
        scaler_x = scalers[lag_size]['scaler_x']
        scaler_y = scalers[lag_size]['scaler_y']
        X_scaled = scaler_x.transform(X)
        
        # Predecir
        model = models[lag_size]
        y_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled)
        
        # Formatear respuesta
        forecast = y_pred[0].tolist()
        
        return PredictionResponse(
            volatility_forecast=forecast,
            horizon_days=7,
            lag_size=lag_size,
            model_info={
                "model_type": "MLP",
                "architecture": str(model.hidden_layer_sizes),
                "activation": model.activation,
                "alpha": model.alpha
            }
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))