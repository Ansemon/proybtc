# 📊 Proyecto: Predicción de Precio y Volatilidad de Bitcoin con Deep Learning

## 🎯 Objetivo

Este proyecto integrador desarrolla un sistema de predicción para el mercado de Bitcoin que combina técnicas avanzadas de deep learning con validación temporal rigurosa. El objetivo principal es predecir el precio de cierre diario y la volatilidad de Bitcoin utilizando exclusivamente datos históricos de precios, aplicando ingeniería de características temporales y redes neuronales MLP.

## 🚀 Contexto y Motivación

El mercado de criptomonedas, particularmente Bitcoin, se caracteriza por su alta volatilidad y comportamiento complejo. Mientras que la mayoría de los esfuerzos se centran en predecir precios, este proyecto adopta un enfoque dual:

- **Predicción de precios**: Anticipar el valor de cierre diario
- **Forecasting de volatilidad**: Predecir la variabilidad del precio, crucial para gestión de riesgo y estrategias de trading

## 🔬 Metodología Implementada

### 📈 Análisis Exploratorio
- Estadísticas descriptivas de precios y retornos logarítmicos
- Visualización de series temporales y análisis de autocorrelación
- Estudio de heterocedasticidad en los retornos

### ⚙️ Ingeniería de Features
- Cálculo de retornos logarítmicos diarios
- Volatilidad histórica mediante desviación estándar móvil
- Generación de retardos temporales (lags) de 7, 14, 21 y 28 días
- Targets multi-step para horizonte de predicción de 7 días

### 🧪 Validación Temporal
- Implementación de GroupKFold para evitar data leakage
- Escalado robusto (fit solo en training en cada fold)
- Evaluación fold-by-fold con métricas múltiples

### 🧠 Modelado con Deep Learning
- Redes MLP multisalida para predicción de 7 días
- Entrenamiento y evaluación por cada configuración de lags
- Análisis exhaustivo de residuos con test BDS

## 📊 Métricas de Evaluación

El proyecto evalúa el desempeño utilizando:
- **MAPE** (Error Porcentual Absoluto Medio)
- **MAE** (Error Absoluto Medio)
- **RMSE** (Raíz del Error Cuadrático Medio)
- **MSE** (Error Cuadrático Medio)
- **Test BDS** para independencia de residuos

## 🛠️ Stack Tecnológico

- **Lenguaje**: Python
- **ML/DL**: TensorFlow/Keras, Scikit-learn
- **Análisis**: Pandas, NumPy, Matplotlib, Seaborn
- **Validación**: Timeseries-CV personalizado
- **Despliegue**: FastAPI, Docker, Tests unitarios

## 📁 Estructura del Proyecto

El proyecto sigue una pipeline completa desde exploración de datos hasta despliegue, incluyendo:
- Análisis exploratorio y preprocesamiento
- Ingeniería de características temporales
- Entrenamiento y validación de modelos
- Evaluación rigurosa y análisis de residuos
- API para predicciones en tiempo real
- Containerización y documentación

## 💡 Valor Añadido

Este trabajo representa un enfoque sistemático y reproducible para el forecasting financiero, con especial énfasis en:
- Validación temporal robusta
- Análisis de residuos exhaustivo
- Comparación múltiple de configuraciones
- Preparación para entornos productivos (MLOps)

El resultado es un sistema capaz de proporcionar predicciones confiables tanto de precio como de volatilidad, esencial para aplicaciones en trading algorítmico y gestión de riesgo.