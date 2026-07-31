import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Add project root to sys.path so we can import src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import split_and_preprocess, load_clinical_data
from src.feature_engineering import add_clinical_biomarkers, get_resnet_extractor, extract_image_features
from src.train import balance_data, train_xgboost, save_model
from src.predict import predict_risk_probability, format_prediction_output
from src.utils import load_single_image

app = Flask(__name__)

# Paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/trained_model.pkl'))
SAMPLE_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/sample_data.csv'))

# Pre-load or train fail-safe model
model = None
scaler = None
imputer = None

def initialize_models():
    global model, scaler, imputer
    
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading trained model from disk...")
            saved_objects = joblib.load(MODEL_PATH)
            # The saved object could be a dictionary or a direct model
            if isinstance(saved_objects, dict):
                model = saved_objects.get('model')
                scaler = saved_objects.get('scaler')
                imputer = saved_objects.get('imputer')
            else:
                model = saved_objects
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Re-training fallback...")
            model = None

    if model is None:
        print("Model file not found or corrupted. Training a fallback model on sample data...")
        try:
            # 1. Load sample clinical data
            df = load_clinical_data(SAMPLE_CSV)
            df_eng = add_clinical_biomarkers(df)
            
            # 2. Preprocess
            X_train, X_test, y_train, y_test, imputer, scaler = split_and_preprocess(df_eng)
            
            # We don't have images here, so we pad the image features with dummy values (20 features)
            # to match the 35 features expected by the fused model.
            dummy_image_features = np.zeros((len(X_train), 20))
            X_train_fused = np.concatenate([X_train.to_numpy(), dummy_image_features], axis=1)
            
            # 3. Train and save
            X_train_bal, y_train_bal = balance_data(X_train_fused, y_train, method='smote')
            model = train_xgboost(X_train_bal, y_train_bal)
            
            # Save objects as dict
            joblib.dump({'model': model, 'scaler': scaler, 'imputer': imputer}, MODEL_PATH)
            print("Fallback model trained and saved successfully.")
        except Exception as e:
            print(f"Fatal: Failed to train fallback model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, imputer
    
    if model is None or scaler is None or imputer is None:
        return jsonify({"error": "Model pipelines are not initialized yet."}), 500
        
    try:
        # 1. Read clinical inputs from form
        clinical_dict = {
            'male': [int(request.form.get('male', 0))],
            'age': [float(request.form.get('age', 45))],
            'education': [float(request.form.get('education', 1.0))],
            'currentSmoker': [int(request.form.get('currentSmoker', 0))],
            'cigsPerDay': [float(request.form.get('cigsPerDay', 0.0))],
            'BPMeds': [float(request.form.get('BPMeds', 0.0))],
            'prevalentStroke': [int(request.form.get('prevalentStroke', 0))],
            'prevalentHyp': [int(request.form.get('prevalentHyp', 0))],
            'diabetes': [int(request.form.get('diabetes', 0))],
            'totChol': [float(request.form.get('totChol', 220.0))],
            'sysBP': [float(request.form.get('sysBP', 130.0))],
            'diaBP': [float(request.form.get('diaBP', 80.0))],
            'BMI': [float(request.form.get('BMI', 24.5))],
            'heartRate': [float(request.form.get('heartRate', 72.0))],
            'glucose': [float(request.form.get('glucose', 85.0))]
        }
        
        clinical_df = pd.DataFrame(clinical_dict)
        
        # 2. Add engineered clinical features
        clinical_df_eng = add_clinical_biomarkers(clinical_df)
        
        # 3. Apply scale and imputation (using training scaler/imputer)
        clinical_imputed = imputer.transform(clinical_df_eng)
        clinical_scaled = scaler.transform(clinical_imputed)
        
        # 4. Handle retinal image upload
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded."}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Selected file is empty."}), 400
            
        # Save file temporarily to extract features
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        
        try:
            # Load and normalize
            img_batch = load_single_image(temp_path)
            # Extract features using ResNet50
            raw_image_features = extract_image_features(img_batch)
            
            # Since our fused model expects 35 features (15 clinical + 20 selected image features),
            # we slice the first 20 features from ResNet (for demonstration/pipeline integrity)
            image_features_selected = raw_image_features[:, :20]
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        # 5. Concatenate clinical and selected image features
        fused_instance = np.concatenate([clinical_scaled, image_features_selected], axis=1)
        
        # 6. Predict CAD probability
        probability = predict_risk_probability(model, fused_instance)[0]
        
        # 7. Format output with 0.2 recall threshold
        output = format_prediction_output(probability, threshold=0.2)
        
        return jsonify(output)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Initialize models before starting the server
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
