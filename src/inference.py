"""
Script de inferencia - Para predecir en datos nuevos.

Uso:
    python src/inference.py --input new_transactions.csv --output predictions.csv
"""
import pandas as pd
import numpy as np
import argparse
from src.save_model import load_model, predict


def preprocess_new_data(df, feature_cols):
    """
    Preprocesar datos nuevos igual que los datos de entrenamiento.
    """
    # Features que necesitamos crear
    if 'Amount_log' not in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])

    if 'Time_hours' not in df.columns and 'Time' in df.columns:
        df['Time_hours'] = df['Time'] / 3600

    if 'is_high_amount' not in df.columns:
        df['is_high_amount'] = (df['Amount'] > 500).astype(int)

    if 'is_very_high_amount' not in df.columns:
        df['is_very_high_amount'] = (df['Amount'] > 1000).astype(int)

    # Seleccionar solo las features que usamos en training
    X = df[feature_cols].values

    return X


def main(input_path, output_path):
    """Pipeline de inferencia."""
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)

    # Cargar modelo
    model, feature_cols, threshold = load_model()

    # Cargar datos nuevos
    print(f"\nCargando datos de: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  - Transacciones: {len(df):,}")

    # Preprocess
    X = preprocess_new_data(df, feature_cols)

    # Predecir
    print("\nPrediciendo...")
    y_pred, y_proba = predict(model, X, threshold=threshold)

    # Agregar resultados al dataframe
    df['is_fraud_predicted'] = y_pred
    df['fraud_probability'] = y_proba

    # Guardar resultados
    df.to_csv(output_path, index=False)
    print(f"\n✓ Resultados guardados en: {output_path}")

    # Resumen
    fraud_count = y_pred.sum()
    fraud_rate = y_pred.mean()
    print(f"""
╔════════════════════════════════════════════════════════╗
║  RESUMEN DE PREDICCIONES                              ║
╠════════════════════════════════════════════════════════╣
║  Total transacciones: {len(df):>10,}                        ║
║  Fraudes detectados:   {fraud_count:>10,}  ({fraud_rate:.2%})             ║
║  Transacciones clean:   {len(df)-fraud_count:>10,}                        ║
╚════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection Inference')
    parser.add_argument('--input', '-i', required=True, help='CSV input file')
    parser.add_argument('--output', '-o', default='predictions.csv', help='CSV output file')

    args = parser.parse_args()
    main(args.input, args.output)