import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df, target_col='TARGET', test_size=0.2, random_state=42):
    y = df[target_col]
    X = df.drop(columns=['SK_ID_CURR', target_col], errors='ignore')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler
