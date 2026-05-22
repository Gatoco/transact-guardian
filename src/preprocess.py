"""
Preprocessing module para el dataset de fraud detection.
Acá hacemos:
1. Temporal split (train/test por Time)
2. Scale Amount
3. Feature engineering básico
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """Cargar el dataset desde CSV"""
    print("Cargando datos...")
    df = pd.read_csv(path)
    print(f"  ✓ {len(df):,} filas cargadas")
    return df


def temporal_split(df, test_size=0.2):
    """
    Separar datos por tiempo (NO aleatorio).

    Esto evita data leakage: usamos datos pasados para predecir futuros.
    """
    print("\n--- Temporal Split ---")

    n = len(df)
    split_idx = int(n * (1 - test_size))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train):,} filas (Time {train['Time'].min():.0f} - {train['Time'].max():.0f})")
    print(f"  Test:  {len(test):,} filas (Time {test['Time'].min():.0f} - {test['Time'].max():.0f})")

    print(f"\n  Fraud rate en TRAIN: {train['Class'].mean():.4%}")
    print(f"  Fraud rate en TEST:  {test['Class'].mean():.4%}")

    return train, test


def scale_amount(train, test):
    """
    Escalar la columna Amount usando StandardScaler.

    StandardScaler: x_scaled = (x - mean) / std
    Así Amount tendrá distribución similar a V1-V28.
    """
    print("\n--- Scaling Amount ---")

    scaler = StandardScaler()

    # fit SOLO en train (nunca en test!)
    train['Amount_scaled'] = scaler.fit_transform(train[['Amount']])
    test['Amount_scaled'] = scaler.transform(test[['Amount']])

    print(f"  Amount_train: mean={train['Amount_scaled'].mean():.4f}, std={train['Amount_scaled'].std():.4f}")
    print(f"  Amount_test:  mean={test['Amount_scaled'].mean():.4f}, std={test['Amount_scaled'].std():.4f}")

    return train, test


def create_features(df):
    """
    Feature engineering básico.

    Creamos features adicionales para enriquecer el modelo.
    """
    print("\n--- Feature Engineering ---")

    # Amount_log: reduce la skewness de distributions muy asimétricas
    # log1p = log(1 + x), safe para valores cercanos a 0
    df['Amount_log'] = np.log1p(df['Amount'])

    # Time_hours: para mejor interpretación
    df['Time_hours'] = df['Time'] / 3600

    # Is high amount: transacciones arriba de $500 (heurístico)
    df['is_high_amount'] = (df['Amount'] > 500).astype(int)

    # Is very high amount: transacciones arriba de $1000
    df['is_very_high_amount'] = (df['Amount'] > 1000).astype(int)

    print(f"  Features creadas: Amount_log, Time_hours, is_high_amount, is_very_high_amount")

    return df


def prepare_features(train, test):
    """
    Preparar las matrices X e y para el modelo.

    X = todas las features
    y = Class (0=legit, 1=fraud)
    """
    print("\n--- Preparando X e y ---")

    # Features que usamos (V1-V28 + Amount_scaled + nuevas)
    feature_cols = [f'V{i}' for i in range(1, 29)] + [
        'Amount_scaled',
        'Amount_log',
        'Time_hours',
        'is_high_amount',
        'is_very_high_amount'
    ]

    X_train = train[feature_cols].values
    y_train = train['Class'].values

    X_test = test[feature_cols].values
    y_test = test['Class'].values

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols


def preprocess_pipeline(csv_path):
    """
    Pipeline completo de preprocessing.

    Returns:
        X_train, y_train, X_test, y_test, feature_cols
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_data(csv_path)

    # 2. Temporal split
    train, test = temporal_split(df, test_size=0.2)

    # 3. Feature engineering (antes de escalar para usar Amount original)
    train = create_features(train)
    test = create_features(test)

    # 4. Scale Amount
    train, test = scale_amount(train, test)

    # 5. Prepare X and y
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(train, test)

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETADO")
    print("=" * 60)

    return X_train, y_train, X_test, y_test, feature_cols


if __name__ == "__main__":
    csv_path = '/home/iwakura/Documentos/github-projects/prachub/build-a-payment-fraud-detection-model/creditcard.csv'
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)