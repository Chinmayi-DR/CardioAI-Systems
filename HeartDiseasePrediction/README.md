# ❤️ CardioAI - Heart Disease Prediction System

CardioAI is a state-of-the-art clinical decision support web application. By training classification machine learning algorithms on physiological attributes and lifestyle variables, the system predicts patient cardiovascular risk levels and recommends tailored precautions.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Installation
Clone or navigate to the project directory:
```bash
cd HeartDiseasePrediction
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Website
Start the local development server:
```bash
python app.py
```
After executing, open your browser and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

> [!NOTE]
> **Out-of-the-Box Execution**: If the synthetic dataset and pickled preprocessor models do not exist on startup, `app.py` automatically triggers `generate_dataset.py` followed by `train_model.py`. The application will start immediately without requiring manual data preparation steps.

---

## 🛠️ System Architecture

### 1. Dataset Generation (`generate_dataset.py`)
Generates a cohort of **10,000 synthetic patient records** using deterministic risk scoring and probabilistic noise distributions:
- **Demographics & Lifestyles**: Age (18-90), Sex (Male/Female), Smoker (Yes/No), Diabetes (Yes/No), BMI (15-45), Family History (Yes/No), Physical Activity (Low/Moderate/High).
- **Vitals**: Resting Blood Pressure (80-200 mmHg), Serum Cholesterol (120-600 mg/dL), Fasting Blood Sugar (0/1), Max Heart Rate (60-210 bpm).
- **Cardiology Diagnostics**: Chest Pain Type (Angina types), Rest ECG (Normal/Abnormalities/LVH), Exercise Induced Angina (Yes/No), Oldpeak ST Depression (0.0-6.5), ST Slope (Upsloping/Flat/Downsloping), Fluoroscopy Vessels (0-4), Thalassemia Scan (Normal/Defects).

Medical rules correlate features to the risk outcome (e.g. older age, high BP, high cholesterol, active smoking, and family history increase risk; physical activity lowers risk).

### 2. Preprocessing & Training (`train_model.py`)
Cleans data, transforms columns, and trains/evaluates four classification models:
- **Preprocessing Pipeline**: One-hot encodes categorical parameters and standardizes numerical variables, exporting `encoder.pkl` and `scaler.pkl`.
- **Algorithms Compared**: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors.
- **Model Selection**: Compares F1-Scores and automatically saves the best-performing model as `model.pkl`.
- **Asset Visualizations**: Generates correlation matrices, ROC curves, confusion matrices, and feature contribution plots saved inside `static/images/` and outputs metrics to JSON inside `static/data/`.

---

## 💻 Web Application Features

1. **Dashboard Home**: High-fidelity hero section with a beating heart animation, clinical context, and general symptoms/risk summaries.
2. **Diagnostic Risk Form**: Interactive entry form for all 18 patient clinical attributes with responsive field bounds and front-end validators.
3. **Cardiovascular Scorecard**: Displays patient classification results (High Risk vs. Low Risk) paired with a fluidly animated SVG risk percentage gauge, model confidence, and medical precautions.
4. **Clinical Dataset Insights**: Visualizes cohort distributions, correlation matrices, demographics progress indicators, and dynamic statistics tables.
5. **Model Performance**: Displays a comparison scorecard comparing trained algorithms, ROC graphs, and feature importances.
6. **Dark & Light Themes**: Persistent color theme mode toggles saved via local storage.
