#!/usr/bin/env python3
"""
Main script - Ejecuta el pipeline completo de fraud detection.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess_pipeline
from src.train_rf import compare_models


def main():
    """Ejecutar pipeline completo."""
    print("=" * 70)
    print("PAYMENT FRAUD DETECTION PIPELINE")
    print("=" * 70)
    print("""
Este script ejecuta:
1. Preprocessing (temporal split, feature engineering, scaling)
2. Entrenamiento de modelos (Logistic Regression y Random Forest)
3. Threshold optimization
4. Comparación de modelos
5. Guardar modelo final
""")

    csv_path = 'creditcard.csv'

    # Run comparison
    model, metrics = compare_models(csv_path)

    print("=" * 70)
    print("✓ PIPELINE COMPLETADO")
    print("=" * 70)
    print("""
Próximos pasos:
- Ver visualizaciones en notebooks/
- Hacer inferencia: python src/inference.py --input data.csv
- Ver README.md para documentación completa
""")


if __name__ == "__main__":
    main()