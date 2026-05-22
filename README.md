# Transact Guardian

> Pipeline de Machine Learning para detección de fraude en transacciones con tarjeta de crédito.

## Prueba Técnica

Este proyecto fue desarrollado como parte de una **prueba técnica** para el rol de **Machine Learning Engineer** en **Adyen**.

### Contexto de la Prueba

**Empresa:** Adyen
**Rol:** Machine Learning Engineer
**Categoría:** Machine Learning
**Dificultad:** Easy
**Tipo de evaluación:** HR Screen

La prueba consistía en:

**Parte 1 - Fundamentos ML:**
- ¿Qué es overfitting?
- ¿Cómo detectar overfitting?
- Regularización L1 vs L2

**Parte 2 - Pipeline Práctico:**
- Construir un pipeline runnable de detección de fraude
- Temporal train/validation/test split sin data leakage
- Manejo de missing values y categorical features
- Address severe class imbalance
- Entrenar un baseline model
- Métricas apropiadas para fraud detection
- Explicar mejoras futuras

---

## Stack Tecnológico

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| **ML** | scikit-learn, Random Forest | Modelo de detección |
| **API** | Flask + Python | Endpoints de predicción |
| **Base de datos** | PostgreSQL | Almacenar transacciones y predicciones |
| **Experiment tracking** | MLflow | Registrar experimentos y métricas |
| **Container** | Docker + docker-compose | Despliegue reproducible |

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE                           │
├─────────────────────────────────────────────────────────────┤
│  Flask API (:5000)  ←→  PostgreSQL (:5432)                  │
│         │                     │                              │
│         └────── MLflow (:5001) ──────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Decisiones Técnicas

### 1. Temporal Split en vez de Random Split

Se separó train/test por tiempo (80/20) para simular el escenario real de producción: entrenar con datos pasados, predecir datos futuros.

**Justificación**: En fraud detection, usar datos futuros para predecir el pasado causa data leakage y no representa la realidad.

### 2. Random Forest sobre Logistic Regression

Se eligió Random Forest como modelo final por:
- Mejor capacidad de capturar relaciones no-lineales
- Feature importance integrada para interpretabilidad
- Robusto al class imbalance con `class_weight='balanced'`

### 3. Class Weights para Manejar Imbalance

El dataset tiene 99.83% transacciones legítimas. Usar `class_weight='balanced'` penaliza más los errores en la clase minoritaria (fraudes), resultando en mejor recall.

### 4. Threshold Optimization

En vez de usar threshold por defecto (0.5), se buscó el threshold que maximiza F1-score usando precision-recall curve.

### 5. API Key Authentication

Se implementó autenticación simple con API key en headers para proteger los endpoints de predicción.

## Dataset

Credit Card Fraud Detection - 284,807 transacciones (0.17% fraudes).

> **Nota:** El dataset no está incluido en el repositorio (143MB > límite de 100MB de GitHub).
> Descárgalo de: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
> y colócalo en la raíz del proyecto como `creditcard.csv`

## Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| **Precision** | 93.55% |
| **Recall** | 77.33% |
| **F1-Score** | 0.8467 |
| **ROC-AUC** | 0.9848 |
| **Fraudes detectados** | 58/75 (77.33%) |

## API Endpoints

| Método | Endpoint | Auth | Descripción |
|--------|----------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/api/v1/predict` | API Key | Predicción single transaction |
| POST | `/api/v1/predict/batch` | API Key | Predicción en batch (max 1000) |
| GET | `/api/v1/predictions` | API Key | Historial de predicciones |
| GET | `/api/v1/predictions/{id}` | API Key | Ver predicción específica |

## Quick Start

### 1. Generar API Key
```bash
python scripts/generate_api_key.py
```

### 2. Levantar servicios
```bash
docker-compose up -d
```

### 3. Hacer predicción
```bash
export API_KEY="fk_tu-api-key-aqui"

curl -X POST http://localhost:5000/api/v1/predict \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.35, "V2": 0.27, "V3": 2.53,
    "amount": 149.62, "time_seconds": 0
  }'
```

### 4. Ver MLflow UI
```
http://localhost:5001
```

## Estructura del Proyecto

```
├── docker-compose.yml          # Orquestación de servicios
├── Dockerfile.flask            # Imagen de Flask API
├── Dockerfile.mlflow           # Imagen de MLflow
├── scripts/
│   ├── init_db.sql             # Schema de PostgreSQL
│   └── generate_api_key.py     # Generador de API key
├── src/
│   ├── api/
│   │   ├── app.py              # Flask app principal
│   │   ├── auth.py             # Middleware de autenticación
│   │   ├── db.py               # Conexión PostgreSQL
│   │   └── models.py           # Endpoints de predicción
│   ├── preprocess.py           # Pipeline de preprocessing
│   ├── train_rf.py             # Script de entrenamiento
│   └── eda.py                  # Análisis exploratorio
├── models/
│   └── fraud_detection_model.pkl  # Modelo pre-entrenado
├── notebooks/                  # Visualizaciones EDA
└── .env                        # Variables de entorno
```

## Autor

Desarrollado como práctica técnica para rol de Machine Learning Engineer.