"""
Generate synthetic payment fraud dataset for demonstration purposes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N_TRANSACTIONS = 100000
FRAUD_RATE = 0.02  # 2% fraud rate

def generate_timestamp(start_date, n_samples):
    """Generate temporal sequence of timestamps"""
    return [start_date + timedelta(minutes=i*2) for i in range(n_samples)]

def generate_dataset(n_samples=N_TRANSACTIONS, fraud_rate=FRAUD_RATE):
    customers = [f'CUST_{i:05d}' for i in range(5000)]
    merchants = [f'MERCH_{i:04d}' for i in range(500)]
    countries = ['US', 'MX', 'BR', 'GB', 'DE', 'FR', 'ES', 'IT', 'JP', 'AU']
    payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet']
    device_types = ['mobile', 'desktop', 'tablet', 'pos_terminal']

    start_date = datetime(2024, 1, 1)
    timestamps = generate_timestamp(start_date, n_samples)

    data = {
        'transaction_id': [f'TX{i:010d}' for i in range(n_samples)],
        'timestamp': timestamps,
        'customer_id': np.random.choice(customers, n_samples),
        'merchant_id': np.random.choice(merchants, n_samples),
        'merchant_category': np.random.choice(['retail', 'groceries', 'travel', 'entertainment',
                                               'restaurants', 'utilities', 'healthcare', 'education'],
                                              n_samples, p=[0.25, 0.20, 0.12, 0.10,
                                                           0.10, 0.08, 0.08, 0.07]),
        'country': np.random.choice(countries, n_samples),
        'payment_method': np.random.choice(payment_methods, n_samples,
                                          p=[0.40, 0.30, 0.15, 0.15]),
        'device_type': np.random.choice(device_types, n_samples,
                                       p=[0.35, 0.30, 0.10, 0.25]),
        'amount': np.random.lognormal(4, 1.5, n_samples),
    }

    df = pd.DataFrame(data)

    df['amount'] = df['amount'].round(2)
    df['customer_country'] = df['country']

    is_fraud = np.random.random(n_samples) < fraud_rate

    high_risk_merchants = np.random.choice(merchants, 50)
    high_risk_countries = ['MX', 'BR']
    high_risk_payment_methods = ['digital_wallet']
    high_risk_devices = ['mobile']

    for idx in df[is_fraud].index:
        if np.random.random() < 0.3:
            df.loc[idx, 'merchant_id'] = np.random.choice(high_risk_merchants)
        if np.random.random() < 0.25:
            df.loc[idx, 'country'] = np.random.choice(high_risk_countries)
        if np.random.random() < 0.2:
            df.loc[idx, 'payment_method'] = np.random.choice(high_risk_payment_methods)
        if np.random.random() < 0.15:
            df.loc[idx, 'device_type'] = np.random.choice(high_risk_devices)

        amount_multiplier = np.random.uniform(2, 5)
        df.loc[idx, 'amount'] = df.loc[idx, 'amount'] * amount_multiplier

        if np.random.random() < 0.1:
            df.loc[idx, 'amount'] = np.random.uniform(1000, 5000)

    is_fraud_pattern = (
        (df['merchant_id'].isin(high_risk_merchants)).astype(int) * 0.2 +
        (df['country'].isin(high_risk_countries)).astype(int) * 0.15 +
        (df['payment_method'].isin(high_risk_payment_methods)).astype(int) * 0.1 +
        (df['amount'] > 500).astype(int) * 0.1
    ) > 0.3

    final_fraud = is_fraud | is_fraud_pattern
    df['is_fraud'] = final_fraud.astype(int)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 6)).astype(int)

    customer_base_tx = df.groupby('customer_id').size()
    df['customer_tx_count'] = df['customer_id'].map(customer_base_tx)

    df['amount_missing'] = np.random.random(n_samples) < 0.01
    df.loc[df['amount_missing'], 'amount'] = np.nan

    df['device_id'] = df.apply(
        lambda x: f"DEV_{hash(x['customer_id'] + x['device_type']) % 100000:05d}"
        if np.random.random() > 0.02 else None, axis=1
    )

    return df

if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    df = generate_dataset()

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSample data:\n{df.head()}")

    output_path = "data/raw/transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")