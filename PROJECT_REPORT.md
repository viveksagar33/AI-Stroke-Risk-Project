# Mini Project Report

## Title
Explainable AI-Based Stroke Risk Prediction and Personalized Preventive Healthcare Recommendation System

## Abstract
The proposed system demonstrates an explainable machine-learning pipeline for estimating stroke risk from demographic and health-related features. Multiple classification algorithms are trained and compared, after which the best-performing model by ROC-AUC is selected. The system provides a probability-like model score, a simple risk band, interpretable contributing factors, and general preventive-health prompts. The web application is implemented using Python and Flask.

## Objectives
1. Build a machine-learning model for stroke-risk estimation.
2. Compare multiple classification algorithms.
3. Improve transparency through explainable factors.
4. Generate general preventive-health prompts based on input risk factors.
5. Provide an easy-to-use web interface.

## Algorithms
- Logistic Regression: interpretable linear baseline.
- Random Forest: ensemble of decision trees.
- Gradient Boosting: sequential tree ensemble.

## XAI approach
The prototype exposes the major factors that were used by a simple transparent explanation layer and can be extended with SHAP for local and global explanations. For a research-grade implementation, SHAP plots, calibration curves, confidence intervals, subgroup fairness metrics, and clinician review should be added.

## Recommendation engine
Recommendations are generated from explicit rules tied to entered factors. This is intentional: it prevents the recommendation text from being an opaque generative model. In a production research system, recommendations should be clinically reviewed and linked to authoritative guidelines.

## Future scope
- Replace synthetic data with a real, licensed dataset.
- Add SHAP waterfall and beeswarm visualizations.
- Add model calibration.
- Add cross-validation and confidence intervals.
- Add fairness evaluation across demographic groups.
- Add secure authentication, audit logs, encryption, and privacy controls.
- Validate recommendations with healthcare professionals.
- Conduct prospective/clinical validation before any real-world use.
