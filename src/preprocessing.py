import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load dataset from CSV.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess tabular data:
    - Drop 'id' if present.
    - Separate features (X) and target (y).
    - Scale features.
    """
    # Drop ID column if exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Identify target column (CLASS_LABEL)
    target_col = 'CLASS_LABEL'
    if target_col not in df.columns:
        # Fallback or error
        print(f"Target column '{target_col}' not found.")
        return None, None

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_scaled = imputer.fit_transform(X_scaled)

    return X_scaled.astype(np.float32), y.astype(np.float32), scaler, imputer
