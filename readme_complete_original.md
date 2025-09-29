# Bitcoin Volatility Prediction API

Sistema de predicción de volatilidad de Bitcoin usando MLP con validación cruzada temporal.

## Instalación

### Desde código fuente

```bash
cd proyectobtc
pip install -r requirements.txt
```

### Desde archivo .tar

```bash
tar -xvf proyectobtc.tar
cd proyectobtc
pip install -r requirements.txt
```

## Uso

### 1. Guardar modelos (primera vez)

```bash
python save_trained_models.py
```

### 2. Ejecutar API

```bash
uvicorn app.api:app --reload
```

La API estará disponible en http://localhost:8000

### 3. Probar

```python
import requests

data = {"lags": [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5]}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

O visitar http://localhost:8000/docs para documentación interactiva.

## Tests

```bash
pytest tests/ -v
```

## Docker

```bash
# Construir
docker build -t btc-volatility-api .

# Ejecutar
docker-compose up -d
```

## Estructura

```
proyectobtc/
├── app/
│   ├── api.py              # FastAPI endpoints
│   ├── utils.py            # Utilidades
│   └── models/             # Modelos entrenados
├── tests/
│   └── test_api.py         # Tests unitarios
├── save_trained_models.py  # Script de entrenamiento
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Endpoints

- `GET /` - Información general
- `GET /health` - Estado de la API
- `POST /predict` - Predecir volatilidad
- `GET /docs` - Documentación Swagger

## Requisitos

- Python 3.9+
- Dataset: `btc_1d_data_2018_to_2025.csv`
