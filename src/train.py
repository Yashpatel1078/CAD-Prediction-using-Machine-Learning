import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

def balance_data(X_train, y_train, method='smote_enn', random_state=42):
    """Handles class imbalances using SMOTE or SMOTEENN hybrid resampling."""
    if method == 'smote_enn':
        resampler = SMOTEENN(random_state=random_state)
    else:
        resampler = SMOTE(random_state=random_state)
    
    X_bal, y_bal = resampler.fit_resample(X_train, y_train)
    return X_bal, y_bal

def train_xgboost(X_train, y_train, random_state=42):
    """Trains a tuned XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def train_stacking_ensemble(X_train, y_train, random_state=42):
    """Trains a stacking ensemble using RF, SVC, KNN as base models and Logistic Regression as meta-learner."""
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')),
        ('svc', SVC(probability=True, random_state=random_state)),
        ('knn', KNeighborsClassifier())
    ]
    meta_learner = LogisticRegression(max_iter=1000)
    
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    stacking_model.fit(X_train, y_train)
    return stacking_model

def save_model(model, filepath='../models/trained_model.pkl'):
    """Saves the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully to {filepath}")

def load_model(filepath='../models/trained_model.pkl'):
    """Loads a trained model from disk."""
    return joblib.load(filepath)
