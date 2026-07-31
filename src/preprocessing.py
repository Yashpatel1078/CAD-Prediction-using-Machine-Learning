import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def load_clinical_data(file_path):
    """Loads CSV and drops rows with missing target values."""
    df = pd.read_csv(file_path)
    # Drop rows if target (TenYearCHD) is null
    df = df.dropna(subset=['TenYearCHD'])
    return df

def clean_data(df):
    """Optionally drops rows with missing education/BMI if simple cleaning is desired."""
    # Drop key columns with tiny missing values, leaving larger ones for imputer
    clean_df = df.dropna(subset=['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate'])
    return clean_df

def split_and_preprocess(df, test_size=0.2, random_state=42):
    """
    Splits clinical data into train/test sets and fits imputer and scaler.
    Ensures ZERO data leakage.
    """
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    
    # Stratified split first
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # KNN Imputation fit on train, transform on test
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    # Scaling fit on train, transform on test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Re-convert to DataFrame with original columns
    cols = X.columns
    X_train_df = pd.DataFrame(X_train_scaled, columns=cols)
    X_test_df = pd.DataFrame(X_test_scaled, columns=cols)
    
    return X_train_df, X_test_df, y_train, y_test, imputer, scaler
