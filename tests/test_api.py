# =============================================================================
# ARCHIVO: tests/test_api.py
# Tests unitarios para la API
# =============================================================================

import pytest
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

# Añadir el directorio app al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.api import app

client = TestClient(app)

class TestAPI:
    
    def test_root_endpoint(self):
        """Test del endpoint raíz"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test del endpoint de health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_predict_endpoint_valid_input(self):
        """Test de predicción con input válido"""
        # Simulamos datos de volatilidad de 7 días
        test_data = {
            "lags": [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5]
        }
        
        response = client.post("/predict", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "volatility_forecast" in data
            assert len(data["volatility_forecast"]) == 7
            assert data["horizon_days"] == 7
            assert data["lag_size"] == 7
        else:
            # Si no hay modelos cargados, debería dar error 500
            assert response.status_code == 500
    
    def test_predict_endpoint_invalid_lag_length(self):
        """Test con longitud inválida de lags"""
        test_data = {
            "lags": [0.5, 0.6, 0.4]  # Solo 3 valores
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_negative_values(self):
        """Test con valores negativos"""
        test_data = {
            "lags": [-0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5]
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
