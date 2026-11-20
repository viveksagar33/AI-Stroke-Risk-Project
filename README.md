# Explainable AI-Based Stroke Risk Prediction and Personalized Preventive Healthcare Recommendation System

## Important
This is an **educational/demo ML project**, not a medical diagnostic or treatment system. Predictions are estimates from the supplied/demo data and must not be used to make clinical decisions.

## Features
- Web interface using Flask
- Three ML models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Automatic model comparison using ROC-AUC
- Explainability using model feature contributions (SHAP is optional)
- Personalized preventive-health suggestions based on risk factors
- Risk probability and Low/Moderate/High demo risk category
- Training metrics and feature importance
- Bundled synthetic demonstration dataset so the project runs immediately

## Project structure
```
stroke_risk_xai_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── stroke_demo.csv
├── models/
│   └── (generated model files)
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Requirements
Python 3.10+ is recommended.

## Run on Windows
1. Extract the ZIP.
2. Open Command Prompt/PowerShell in the project folder.
3. Create a virtual environment:
   `python -m venv venv`
4. Activate it:
   `venv\Scripts\activate`
5. Install packages:
   `pip install -r requirements.txt`
6. Train the models:
   `python train_model.py`
7. Start the web application:
   `python app.py`
8. Open:
   `http://127.0.0.1:5000`

## Dataset
The included CSV is synthetic demonstration data designed only to make the application executable without an external download. For an academic evaluation, replace it with a properly licensed real stroke dataset and document its source, preprocessing, ethics, and limitations.

Expected columns:
`age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status, gender, ever_married, work_type, Residence_type, stroke`

## Explainability
The app shows:
- top positive contributors to the model prediction
- top negative contributors
- a human-readable explanation of the major factors

If `shap` is installed, you can extend the project to use SHAP plots. The current implementation deliberately uses model-native contribution calculations so the demo has fewer dependency issues.

## Academic presentation
Suggested modules:
1. Data collection
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model comparison
6. Explainable AI
7. Personalized recommendations
8. Web deployment
9. Evaluation and limitations

## Limitations
- Synthetic dataset is not suitable for clinical validation.
- Risk categories are demonstration thresholds.
- Recommendations are general wellness prompts, not personalized medical advice.
- Model performance can be biased by dataset quality and population differences.
- Real deployment requires clinical validation, calibration, privacy/security controls, fairness testing, monitoring, and regulatory review.
