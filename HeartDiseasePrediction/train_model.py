import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

def train_and_evaluate():
    # Make sure output directories exist
    os.makedirs(os.path.join('static', 'images'), exist_ok=True)
    os.makedirs(os.path.join('static', 'data'), exist_ok=True)
    
    csv_path = os.path.join('dataset', 'heart_disease_dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_dataset.py first.")
        
    df = pd.read_csv(csv_path)
    
    # 1. Basic cleaning
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.dropna()
    
    # Save dataset stats for Insights page
    stats = {
        'shape': df_cleaned.shape,
        'columns': list(df_cleaned.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicates_removed': int(len(df) - len(df_cleaned)),
        'heart_disease_counts': df_cleaned['heart_disease'].value_counts().to_dict(),
        'sex_counts': df_cleaned['sex'].value_counts().to_dict(),
        'smoker_counts': df_cleaned['smoker'].value_counts().to_dict(),
        'diabetes_counts': df_cleaned['diabetes'].value_counts().to_dict(),
        'family_history_counts': df_cleaned['family_history'].value_counts().to_dict(),
        'summary_statistics': df_cleaned.describe().to_dict()
    }
    
    with open(os.path.join('static', 'data', 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    # Generate static plots for Dataset Insights
    print("Generating Dataset Insights plots...")
    # Plot 1: Correlation Heatmap (numerical columns only)
    plt.figure(figsize=(10, 8))
    numeric_cols = ['age', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'max_heart_rate', 'oldpeak', 'major_vessels', 'bmi', 'heart_disease']
    sns.heatmap(df_cleaned[numeric_cols].corr(), annot=True, cmap='RdBu', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap (Numerical Features & Target)')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot 2: Distibution of Age vs Heart Disease
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_cleaned, x='age', hue='heart_disease', multiple='stack', palette=['#2ecc71', '#e74c3c'], bins=20)
    plt.title('Age Distribution by Heart Disease Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(title='Heart Disease', labels=['High Risk', 'Low Risk'])
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'age_distribution.png'), dpi=150)
    plt.close()

    # Plot 3: Risk by Chest Pain Type
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_cleaned, x='chest_pain_type', hue='heart_disease', palette=['#2ecc71', '#e74c3c'])
    plt.title('Heart Disease Risk by Chest Pain Type')
    plt.xlabel('Chest Pain Type')
    plt.ylabel('Count')
    plt.legend(title='Heart Disease', labels=['High Risk', 'Low Risk'])
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'chest_pain_risk.png'), dpi=150)
    plt.close()

    # Define features and target
    X_raw = df_cleaned.drop(columns=['heart_disease'])
    y = df_cleaned['heart_disease']
    
    categorical_cols = ['sex', 'chest_pain_type', 'rest_ecg', 'exercise_induced_angina', 'slope', 'thal', 'smoker', 'diabetes', 'family_history', 'physical_activity']
    numerical_cols = ['age', 'resting_bp', 'cholesterol', 'fasting_blood_sugar', 'max_heart_rate', 'oldpeak', 'major_vessels', 'bmi']
    
    # Fit & save preprocessors
    print("Fitting and saving preprocessors...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(X_raw[categorical_cols])
    joblib.dump(encoder, 'encoder.pkl')
    
    scaler = StandardScaler()
    scaler.fit(X_raw[numerical_cols])
    joblib.dump(scaler, 'scaler.pkl')
    
    # Transform data
    X_cat_encoded = pd.DataFrame(encoder.transform(X_raw[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    X_num_scaled = pd.DataFrame(scaler.transform(X_raw[numerical_cols]), columns=numerical_cols)
    X = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=9)
    }
    
    results = {}
    roc_data = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # KNN might not have predict_proba in rare custom scikit-learn configs, but standard KNN does
            y_prob = model.predict_proba(X_test)[:, 1]
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1 Score': float(f1),
            'ROC AUC': float(roc_auc)
        }
        
        # Save ROC Curve details
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    
    # Finish and save ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'roc_curve.png'), dpi=150)
    plt.close()
    
    # Determine the best model based on F1 Score
    best_model_name = max(results, key=lambda k: results[k]['F1 Score'])
    best_model = models[best_model_name]
    
    print(f"\nBest model selected: {best_model_name} with F1-Score: {results[best_model_name]['F1 Score']:.4f}")
    joblib.dump(best_model, 'model.pkl')
    
    # Save confusion matrix for best model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    plt.ylabel('Actual Risk')
    plt.xlabel('Predicted Risk')
    plt.title(f'Confusion Matrix ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Save Feature Importance for the best model (or alternative if KNN/LR coefficients)
    plt.figure(figsize=(10, 6))
    if best_model_name in ['Random Forest', 'Decision Tree']:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        plt.barh(range(len(indices)), importances[indices], align='center', color='#3498db')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title(f'Top 10 Feature Importances ({best_model_name})')
    else:
        # For Logistic Regression coefficients
        if best_model_name == 'Logistic Regression':
            coefs = best_model.coef_[0]
            indices = np.argsort(np.abs(coefs))[-10:] # Top 10
            plt.barh(range(len(indices)), coefs[indices], align='center', color='#3498db')
            plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
            plt.xlabel('Coefficient Value (Direction of Risk)')
            plt.title(f'Top 10 Feature Coefficients ({best_model_name})')
        else:
            # KNN doesn't have intrinsic feature importance, use Random Forest importances as a general proxy
            rf = models['Random Forest']
            importances = rf.feature_importances_
            indices = np.argsort(importances)[-10:]
            plt.barh(range(len(indices)), importances[indices], align='center', color='#e74c3c')
            plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
            plt.xlabel('Relative Importance (RF Proxy)')
            plt.title(f'Top 10 Feature Importances (Random Forest Proxy)')
            
    plt.tight_layout()
    plt.savefig(os.path.join('static', 'images', 'feature_importance.png'), dpi=150)
    plt.close()
    
    # Save performance metadata
    performance_metadata = {
        'best_model_name': best_model_name,
        'results': results
    }
    with open(os.path.join('static', 'data', 'metrics.json'), 'w') as f:
        json.dump(performance_metadata, f, indent=4)
        
    print("Model training and evaluation successfully completed!")

if __name__ == '__main__':
    train_and_evaluate()
