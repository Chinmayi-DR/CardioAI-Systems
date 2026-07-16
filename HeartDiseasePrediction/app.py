import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Core files to check on startup
DATASET_PATH = os.path.join('dataset', 'heart_disease_dataset.csv')
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'encoder.pkl'
STATS_PATH = os.path.join('static', 'data', 'dataset_stats.json')
METRICS_PATH = os.path.join('static', 'data', 'metrics.json')

def check_and_initialize_system():
    """Ensure dataset and model are present on startup. If not, generate and train."""
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Generating dataset dynamically...")
        from generate_dataset import generate_heart_disease_data
        df = generate_heart_disease_data(10000)
        os.makedirs('dataset', exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        print("Dataset generated successfully.")
        
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
        print("Model or preprocessors not found. Training models dynamically...")
        from train_model import train_and_evaluate
        train_and_evaluate()
        print("Model trained and preprocessors saved successfully.")

# Initialize dataset and model prior to serving requests
check_and_initialize_system()

# Load model and preprocessors
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# Helper function to read stats
def get_dataset_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as f:
            return json.load(f)
    return {}

# Helper function to read metrics
def get_model_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return {}

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        try:
            input_data = {
                'age': int(request.form.get('age')),
                'sex': request.form.get('sex'),
                'chest_pain_type': request.form.get('chest_pain_type'),
                'resting_bp': int(request.form.get('resting_bp')),
                'cholesterol': int(request.form.get('cholesterol')),
                'fasting_blood_sugar': int(request.form.get('fasting_blood_sugar')),
                'rest_ecg': request.form.get('rest_ecg'),
                'max_heart_rate': int(request.form.get('max_heart_rate')),
                'exercise_induced_angina': request.form.get('exercise_induced_angina'),
                'oldpeak': float(request.form.get('oldpeak')),
                'slope': request.form.get('slope'),
                'major_vessels': int(request.form.get('major_vessels')),
                'thal': request.form.get('thal'),
                'smoker': request.form.get('smoker'),
                'diabetes': request.form.get('diabetes'),
                'bmi': float(request.form.get('bmi')),
                'family_history': request.form.get('family_history'),
                'physical_activity': request.form.get('physical_activity')
            }
            
            # Prepare inputs as a pandas DataFrame
            input_df = pd.DataFrame([input_data])
            
            categorical_cols = ['sex', 'chest_pain_type', 'rest_ecg', 'exercise_induced_angina', 'slope', 'thal', 'smoker', 'diabetes', 'family_history', 'physical_activity']
            numerical_cols = ['age', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'max_heart_rate', 'oldpeak', 'major_vessels', 'bmi']
            
            # Transform input data
            input_cat_encoded = pd.DataFrame(
                encoder.transform(input_df[categorical_cols]), 
                columns=encoder.get_feature_names_out(categorical_cols)
            )
            input_num_scaled = pd.DataFrame(
                scaler.transform(input_df[numerical_cols]), 
                columns=numerical_cols
            )
            
            processed_input = pd.concat([input_num_scaled, input_cat_encoded], axis=1)
            
            # Run prediction
            risk_class = int(model.predict(processed_input)[0])
            probabilities = model.predict_proba(processed_input)[0]
            risk_probability = float(probabilities[1]) * 100
            
            # Calculate a confidence factor (optional display)
            confidence = float(probabilities[risk_class]) * 100
            
            # Medical advice / Recommendations based on risk classification
            if risk_class == 1:
                # High Risk Recommendations
                lifestyle_advice = [
                    "Consult a cardiologist immediately for a detailed medical evaluation and cardiac screening.",
                    "Monitor your blood pressure and cholesterol levels regularly as advised by your healthcare provider.",
                    "Adopt a low-sodium, heart-healthy diet (like the DASH or Mediterranean diet), rich in whole grains, vegetables, and lean proteins.",
                    "Avoid high-stress activities and smoking completely. Limit alcohol consumption.",
                    "Consult your physician before starting any physical exercise regime."
                ]
                precautions = [
                    "Keep emergency contact numbers handy.",
                    "If you experience chest pain, breathlessness, or left-arm numbness, seek emergency medical care immediately.",
                    "Adhere strictly to any prescribed cardiovascular medications and attend follow-up appointments."
                ]
            else:
                # Low Risk Advice
                lifestyle_advice = [
                    "Maintain a balanced diet rich in fiber, antioxidants, and healthy fats (e.g., olive oil, nuts).",
                    "Engage in at least 150 minutes of moderate-intensity aerobic exercise (like brisk walking or swimming) weekly.",
                    "Maintain a healthy body weight (target BMI between 18.5 and 24.9).",
                    "Continue avoiding tobacco products and limit intake of processed foods and added sugars.",
                    "Manage daily stress through mindfulness, meditation, or regular recreational activities."
                ]
                precautions = [
                    "Conduct annual cardiovascular health checkups to monitor trends as you age.",
                    "Keep tabs on your key metrics: Blood Pressure, Cholesterol, and Blood Glucose levels."
                ]
                
            return render_template(
                'result.html', 
                input_data=input_data,
                risk_class=risk_class,
                risk_probability=round(risk_probability, 1),
                confidence=round(confidence, 1),
                lifestyle_advice=lifestyle_advice,
                precautions=precautions
            )
            
        except Exception as e:
            return render_template('predict.html', error=f"Invalid data entered: {str(e)}")
            
    return render_template('predict.html')

@app.route('/insights')
def insights():
    stats = get_dataset_stats()
    return render_template('insights.html', stats=stats)

@app.route('/performance')
def performance():
    metrics = get_model_metrics()
    return render_template('performance.html', metrics=metrics)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        # Simulate successful submission with message toast
        success_msg = f"Thank you, {name}! Your message has been received. Our team will get back to you at {email} soon."
        return render_template('contact.html', success_msg=success_msg)
    return render_template('contact.html')

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
