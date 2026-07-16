import os
import numpy as np
import pandas as pd

def generate_heart_disease_data(n_samples=10000, seed=42):
    np.random.seed(seed)
    
    # Generate independent features
    age = np.random.randint(18, 91, size=n_samples)
    sex = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
    
    chest_pain_type = np.random.choice(
        ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
        size=n_samples,
        p=[0.15, 0.20, 0.25, 0.40]
    )
    
    resting_bp = np.random.randint(80, 201, size=n_samples)
    cholesterol = np.random.randint(120, 601, size=n_samples)
    fasting_blood_sugar = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    rest_ecg = np.random.choice(
        ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'],
        size=n_samples,
        p=[0.55, 0.30, 0.15]
    )
    
    max_heart_rate = np.random.randint(60, 211, size=n_samples)
    exercise_induced_angina = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.35, 0.65])
    oldpeak = np.round(np.random.uniform(0.0, 6.5, size=n_samples), 1)
    
    slope = np.random.choice(
        ['Upsloping', 'Flat', 'Downsloping'],
        size=n_samples,
        p=[0.35, 0.50, 0.15]
    )
    
    major_vessels = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.55, 0.20, 0.12, 0.08, 0.05])
    
    thal = np.random.choice(
        ['Normal', 'Fixed Defect', 'Reversible Defect'],
        size=n_samples,
        p=[0.55, 0.10, 0.35]
    )
    
    smoker = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])
    diabetes = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.15, 0.85])
    bmi = np.round(np.random.uniform(15.0, 45.0, size=n_samples), 1)
    family_history = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.25, 0.75])
    
    physical_activity = np.random.choice(
        ['Low', 'Moderate', 'High'],
        size=n_samples,
        p=[0.40, 0.40, 0.20]
    )
    
    # Calculate heart disease risk score based on medical correlations
    score = np.zeros(n_samples)
    
    # Age: higher age -> higher risk
    score += (age - 18) / 72.0 * 1.5
    
    # Sex: Male -> higher risk
    score += np.where(sex == 'Male', 0.4, 0.0)
    
    # Chest Pain Type: Typical Angina and Atypical Angina represent higher risk
    score += np.where(chest_pain_type == 'Typical Angina', 1.2, 0.0)
    score += np.where(chest_pain_type == 'Atypical Angina', 0.8, 0.0)
    score += np.where(chest_pain_type == 'Non-anginal Pain', 0.3, 0.0)
    
    # Resting BP: High BP -> higher risk
    score += (resting_bp - 80) / 120.0 * 0.8
    
    # Cholesterol: High Cholesterol -> higher risk
    score += (cholesterol - 120) / 480.0 * 0.8
    
    # Fasting Blood Sugar: High -> higher risk
    score += np.where(fasting_blood_sugar == 1, 0.3, 0.0)
    
    # Rest ECG
    score += np.where(rest_ecg == 'Left Ventricular Hypertrophy', 0.6, 0.0)
    score += np.where(rest_ecg == 'ST-T Abnormality', 0.4, 0.0)
    
    # Max Heart Rate: Lower max heart rate -> higher risk
    score += (210 - max_heart_rate) / 150.0 * 0.6
    
    # Exercise Induced Angina: Yes -> higher risk
    score += np.where(exercise_induced_angina == 'Yes', 1.0, 0.0)
    
    # Oldpeak: Higher ST depression -> higher risk
    score += (oldpeak / 6.5) * 1.5
    
    # Slope: Flat and Downsloping -> higher risk
    score += np.where(slope == 'Downsloping', 0.8, 0.0)
    score += np.where(slope == 'Flat', 0.5, 0.0)
    
    # Major Vessels: More colored vessels -> higher risk
    score += (major_vessels / 4.0) * 1.2
    
    # Thal: Defects -> higher risk
    score += np.where(thal == 'Reversible Defect', 1.2, 0.0)
    score += np.where(thal == 'Fixed Defect', 0.6, 0.0)
    
    # Lifestyle factors
    score += np.where(smoker == 'Yes', 0.8, 0.0)
    score += np.where(diabetes == 'Yes', 0.6, 0.0)
    score += (bmi - 15.0) / 30.0 * 0.6
    score += np.where(family_history == 'Yes', 0.8, 0.0)
    
    # Physical activity: High -> lowers risk, Low -> increases risk
    score += np.where(physical_activity == 'Low', 0.4, 0.0)
    score -= np.where(physical_activity == 'High', 0.4, 0.0)
    
    # Map score to probability using sigmoid function
    # Let's adjust offset to make the heart disease target roughly balanced
    prob = 1.0 / (1.0 + np.exp(-(score - 5.8) / 1.5))
    
    # Add random noise to make the classification boundary realistic and probabilistic
    noise = np.random.normal(0, 0.05, size=n_samples)
    final_prob = np.clip(prob + noise, 0.0, 1.0)
    
    heart_disease = (final_prob >= 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'rest_ecg': rest_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'oldpeak': oldpeak,
        'slope': slope,
        'major_vessels': major_vessels,
        'thal': thal,
        'smoker': smoker,
        'diabetes': diabetes,
        'bmi': bmi,
        'family_history': family_history,
        'physical_activity': physical_activity,
        'heart_disease': heart_disease
    })
    
    return df

if __name__ == '__main__':
    print("Generating synthetic heart disease dataset...")
    df = generate_heart_disease_data(10000)
    
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)
    
    csv_path = os.path.join('dataset', 'heart_disease_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved successfully with {len(df)} rows and {len(df.columns)} columns at {csv_path}")
    print(f"Heart disease class distribution:\n{df['heart_disease'].value_counts(normalize=True)}")
