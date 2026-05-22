"""
EDA - Exploratory Data Analysis
Vamos a entender los datos paso a paso.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config para gráficos bonitos
plt.style.use('default')
sns.set_theme()

def load_data():
    """Cargar el dataset"""
    print("=" * 60)
    print("PASO 1: Cargando datos")
    print("=" * 60)

    df = pd.read_csv('/home/iwakura/Documentos/github-projects/prachub/build-a-payment-fraud-detection-model/creditcard.csv')

    print(f"\n✓ Dataset cargado")
    print(f"  - Filas: {len(df):,}")
    print(f"  - Columnas: {len(df.columns)}")

    return df


def basic_info(df):
    """Información básica del dataset"""
    print("\n" + "=" * 60)
    print("PASO 2: Info básica del dataset")
    print("=" * 60)

    print("\n--- Columnas ---")
    print(df.columns.tolist())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Primeras 5 filas ---")
    print(df.head())


def fraud_analysis(df):
    """Análisis de la variable objetivo"""
    print("\n" + "=" * 60)
    print("PASO 3: Análisis de FRAUDE (variable 'Class')")
    print("=" * 60)

    fraud_counts = df['Class'].value_counts()

    print("\n--- Conteo de clases ---")
    print(f"  Legítimas (0): {fraud_counts[0]:,} ({fraud_counts[0]/len(df)*100:.2f}%)")
    print(f"  Fraudes (1):    {fraud_counts[1]:,} ({fraud_counts[1]/len(df)*100:.4f}%)")

    print("\n⚠️ ESTE ES UN DATASET MUY IMBALANCEADO")
    print(f"   Ratio: {fraud_counts[0]/fraud_counts[1]:.1f}:1")
    print(f"   Por cada fraude, hay {fraud_counts[0]/fraud_counts[1]:.0f} transacciones legítimas")


def amount_analysis(df):
    """Análisis de la columna Amount"""
    print("\n" + "=" * 60)
    print("PASO 4: Análisis de AMOUNT (monto de transacción)")
    print("=" * 60)

    legit = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']

    print("\n--- Transacciones LEGÍTIMAS ---")
    print(f"  Promedio:  ${legit.mean():.2f}")
    print(f"  Mediana:   ${legit.median():.2f}")
    print(f"  Mínimo:    ${legit.min():.2f}")
    print(f"  Máximo:    ${legit.max():.2f}")

    print("\n--- Transacciones FRAUDULENTAS ---")
    print(f"  Promedio:  ${fraud.mean():.2f}")
    print(f"  Mediana:   ${fraud.median():.2f}")
    print(f"  Mínimo:    ${fraud.min():.2f}")
    print(f"  Máximo:    ${fraud.max():.2f}")

    print("\n💡 INSIGHT: Los fraudes tienen un monto PROMEDIO más alto")
    print(f"   Fraudes son {fraud.mean()/legit.mean():.1f}x más altos en promedio")


def time_analysis(df):
    """Análisis de la columna Time"""
    print("\n" + "=" * 60)
    print("PASO 5: Análisis de TIME")
    print("=" * 60)

    print(f"\n--- Rango de tiempo ---")
    print(f"  Tiempo mínimo: {df['Time'].min():,.0f} segundos")
    print(f"  Tiempo máximo: {df['Time'].max():,.0f} segundos")
    print(f"  Duración total: {(df['Time'].max() - df['Time'].min()) / 3600:.1f} horas")

    print("\n💡 INSIGHT: Time representa segundos desde la primera transacción")


def missing_values(df):
    """Valores faltantes"""
    print("\n" + "=" * 60)
    print("PASO 6: Missing Values (valores faltantes)")
    print("=" * 60)

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("\n✓ NO hay valores faltantes en el dataset")
    else:
        print(f"\n--- Total missing: {total_missing} ---")
        print(missing[missing > 0])


def v_columns_analysis(df):
    """Análisis de las columnas V1-V28"""
    print("\n" + "=" * 60)
    print("PASO 7: Columnas V1-V28 (features anónimas)")
    print("=" * 60)

    v_cols = [f'V{i}' for i in range(1, 29)]

    print("\n--- Stats básicas de columnas V ---")
    print(df[v_cols].describe().T[['mean', 'std', 'min', 'max']].head(10))

    print("\n💡 INSIGHT: V1-V28 son componentes de PCA (ya transformados)")
    print("   No necesitas hacer scaling pesado, ya vienen normalizados")


def create_visualizations(df):
    """Crear visualizaciones"""
    print("\n" + "=" * 60)
    print("PASO 8: Creando visualizaciones...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Fraud distribution (pie chart)
    ax1 = axes[0, 0]
    fraud_counts = df['Class'].value_counts()
    ax1.pie([fraud_counts[0], fraud_counts[1]],
            labels=['Legitimate', 'Fraud'],
            autopct='%1.3f%%',
            colors=['lightgreen', 'red'])
    ax1.set_title('Class Distribution (Imbalance)')

    # 2. Amount distribution by class
    ax2 = axes[0, 1]
    legit = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']
    ax2.hist(legit, bins=50, alpha=0.5, label='Legitimate', color='green')
    ax2.hist(fraud, bins=50, alpha=0.5, label='Fraud', color='red')
    ax2.set_xlabel('Amount')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Amount Distribution by Class')
    ax2.legend()
    ax2.set_xlim(0, 2500)

    # 3. Amount boxplot by class
    ax3 = axes[1, 0]
    df.boxplot(column='Amount', by='Class', ax=ax3)
    ax3.set_xlabel('Class (0=Legit, 1=Fraud)')
    ax3.set_ylabel('Amount')
    ax3.set_title('Amount by Class')
    plt.suptitle('')

    # 4. Time distribution
    ax4 = axes[1, 1]
    ax4.hist(df['Time'], bins=50, alpha=0.7, color='blue')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Transaction Time Distribution')

    plt.tight_layout()
    plt.savefig('notebooks/01_eda_visualizations.png', dpi=100)
    print("✓ Guardado: notebooks/01_eda_visualizations.png")
    plt.close()

    # Correlation heatmap (sample for speed)
    print("✓ Creando heatmap de correlación (puede tardar)...")

    # Sample for correlation
    df_sample = df.sample(10000, random_state=42)
    corr = df_sample.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=0)
    plt.title('Correlation Heatmap (sample 10k)')
    plt.tight_layout()
    plt.savefig('notebooks/02_correlation_heatmap.png', dpi=100)
    print("✓ Guardado: notebooks/02_correlation_heatmap.png")
    plt.close()


def summary():
    """Resumen final"""
    print("\n" + "=" * 60)
    print("RESUMEN DEL EDA")
    print("=" * 60)

    print("""
KEY FINDINGS:
1. Dataset MUY imbalanceado: 0.17% fraudes
2. NO hay missing values
3. V1-V28 ya están normalizados (PCA)
4. Amount es la única feature "cruda" que我们需要 escalar
5. Fraudes tienden a tener montos más altos

PRÓXIMOS PASOS:
1. Train/Test Split (temporal para evitar leakage)
2. Escalar Amount (StandardScaler)
3. Entrenar modelo con class_weight='balanced'
4. Evaluar con ROC-AUC, PR-AUC, F1-Score

¿Preguntas hasta aquí?
""")


if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    fraud_analysis(df)
    amount_analysis(df)
    time_analysis(df)
    missing_values(df)
    v_columns_analysis(df)
    create_visualizations(df)
    summary()