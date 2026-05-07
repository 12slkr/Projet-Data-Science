import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

SEED = 42

CAT_COLS = [
    'gender', 'customer_segment', 'signup_channel',
    'contract_type', 'payment_method', 'complaint_type', 'survey_response'
]
# country et city exclus (trop forte cardinalité) ; customer_id exclu (identifiant)
DROP_COLS = ['customer_id', 'country', 'city']


def load_data(path="data/customer_churn_business_dataset.csv"):
    df = pd.read_csv(path)

    # Valeurs manquantes : 2045 NULL dans complaint_type
    df['complaint_type'] = df['complaint_type'].fillna('Unknown')

    # Encodage binaire Yes/No
    df['discount_applied'] = df['discount_applied'].map({'Yes': 1, 'No': 0})
    df['price_increase_last_3m'] = df['price_increase_last_3m'].map({'Yes': 1, 'No': 0})

    # Feature engineering
    df['tickets_per_tenure'] = df['support_tickets'] / (df['tenure_months'] + 1)
    df['avg_monthly_revenue'] = df['total_revenue'] / (df['tenure_months'] + 1)
    df['financial_risk_flag'] = (
        (df['payment_failures'] > 0) & (df['price_increase_last_3m'] == 1)
    ).astype(int)

    return df


def get_features_target(df):
    TARGET = 'churn'
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    return X, y


def get_column_lists(X):
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return num_cols, CAT_COLS


def build_preprocessor(num_cols, cat_cols):
    """Pipeline sklearn anti-leakage. Le fit() doit être appelé UNIQUEMENT sur X_train."""
    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])
    return preprocessor


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y  # indispensable avec le déséquilibre de classes
    )
