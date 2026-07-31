import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2

def add_clinical_biomarkers(df):
    """
    Computes custom cardiac risk metrics from clinical measurements:
    - Pulse Pressure
    - Mean Arterial Pressure (MAP)
    - Life Smoking Intensity Index
    """
    new_df = df.copy()
    
    # Pulse pressure (difference between systolic and diastolic BP)
    new_df['pulse_pressure'] = new_df['sysBP'] - new_df['diaBP']
    
    # Mean arterial pressure (average pressure in arteries during cardiac cycle)
    new_df['MAP'] = new_df['diaBP'] + (new_df['sysBP'] - new_df['diaBP']) / 3.0
    
    # Smoking index (proxy for lifetime smoke exposure)
    new_df['smoke_age_index'] = new_df['cigsPerDay'] * new_df['age']
    
    return new_df

# Global model container for extractor to avoid reloading it constantly
_resnet_model = None

def get_resnet_extractor():
    global _resnet_model
    if _resnet_model is None:
        # Load pre-trained ResNet50 without classification head
        _resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return _resnet_model

def extract_image_features(images_batch):
    """
    Feeds a batch of normalized images to ResNet50 and extracts 2048-dimensional features.
    Assumes images are numpy arrays scaled in [0, 1].
    """
    model = get_resnet_extractor()
    # Rescale to 0-255 for ResNet preprocessing
    rescaled = images_batch * 255.0
    preprocessed = preprocess_input(rescaled)
    
    # Extract deep features
    features = model.predict(preprocessed)
    return features
