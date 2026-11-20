import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "stroke_demo.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA)

target = "stroke"
X = df.drop(columns=[target])
y = df[target]

numeric = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
categorical = ["smoking_status", "gender", "ever_married", "work_type", "Residence_type"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=350, random_state=42, class_weight="balanced", max_depth=10
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=180)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

results = {}
trained = {}

for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4)
    }
    trained[name] = pipe

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = trained[best_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)

# Save transformed feature names and model-native importances where available.
prep = best_model.named_steps["prep"]
model = best_model.named_steps["model"]
feature_names = list(prep.get_feature_names_out())

importance = {}
if hasattr(model, "coef_"):
    vals = np.abs(model.coef_[0])
    importance = dict(sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)[:15])
elif hasattr(model, "feature_importances_"):
    vals = model.feature_importances_
    importance = dict(sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)[:15])

with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
    json.dump({k: float(v) for k, v in importance.items()}, f, indent=2)

print("Training complete.")
print("Best model:", best_name)
print(json.dumps(results, indent=2))
