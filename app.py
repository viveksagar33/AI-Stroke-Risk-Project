from flask import Flask, render_template, request
import os
import json
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(BASE, "models", "best_model.joblib"))

with open(os.path.join(BASE, "models", "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)

def risk_band(p):
    if p < 0.25:
        return "Low"
    if p < 0.55:
        return "Moderate"
    return "High"

def recommendations(d):
    rec = []

    if d["hypertension"] == 1:
        rec.append("Blood-pressure management is important. Track readings and discuss persistent high readings with a qualified clinician.")
    if d["heart_disease"] == 1:
        rec.append("Because heart disease is reported, keep regular follow-up with your healthcare professional and follow prescribed care.")
    if d["avg_glucose_level"] >= 140:
        rec.append("Glucose is elevated in this demo input. Consider appropriate glucose screening and professional guidance.")
    if d["bmi"] >= 25:
        rec.append("A healthy eating pattern and regular physical activity may support healthy weight management.")
    if d["smoking_status"] == "smokes":
        rec.append("Smoking cessation can substantially improve cardiovascular health; consider a structured cessation program.")
    elif d["smoking_status"] == "formerly smoked":
        rec.append("Staying smoke-free is beneficial for long-term cardiovascular health.")
    if d["age"] >= 60:
        rec.append("Age is a non-modifiable risk factor; regular preventive check-ups become increasingly valuable.")
    if not rec:
        rec.append("Maintain a balanced diet, regular physical activity, healthy sleep, and routine preventive check-ups.")

    rec.append("If sudden facial drooping, arm weakness, speech difficulty, severe dizziness, or other possible stroke symptoms occur, seek emergency medical care immediately.")
    return rec

def explain_input(d):
    factors = []
    if d["age"] >= 60: factors.append(("Age", "higher age can be associated with increased stroke risk"))
    if d["hypertension"] == 1: factors.append(("Hypertension", "reported hypertension is a major cardiovascular risk factor"))
    if d["heart_disease"] == 1: factors.append(("Heart disease", "reported heart disease can be associated with elevated stroke risk"))
    if d["avg_glucose_level"] >= 140: factors.append(("Glucose", "the entered glucose value is elevated in this demo"))
    if d["bmi"] >= 25: factors.append(("BMI", "the entered BMI is above the conventional healthy range"))
    if d["smoking_status"] == "smokes": factors.append(("Smoking", "current smoking can increase cardiovascular risk"))
    if not factors:
        factors.append(("Profile", "no major positive demo risk contributors were triggered by the simple explanation rules"))
    return factors[:6]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            d = {
                "age": float(request.form["age"]),
                "hypertension": int(request.form["hypertension"]),
                "heart_disease": int(request.form["heart_disease"]),
                "avg_glucose_level": float(request.form["avg_glucose_level"]),
                "bmi": float(request.form["bmi"]),
                "smoking_status": request.form["smoking_status"],
                "gender": request.form["gender"],
                "ever_married": request.form["ever_married"],
                "work_type": request.form["work_type"],
                "Residence_type": request.form["Residence_type"]
            }

            X = pd.DataFrame([d])
            probability = float(MODEL.predict_proba(X)[0, 1])
            band = risk_band(probability)

            result = {
                "probability": round(probability * 100, 2),
                "band": band,
                "recommendations": recommendations(d),
                "explanations": explain_input(d),
                "model": METRICS["best_model"]
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, metrics=METRICS)

if __name__ == "__main__":
    app.run(debug=True)
