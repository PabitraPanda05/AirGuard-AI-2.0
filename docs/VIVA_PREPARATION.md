# AirGuard AI 2.0 — Viva Preparation Notes

## Project objective
To analyze air-pollution observations, classify AQI categories from pollutant AQI sub-index values, identify unusual pollutant patterns, explain model decisions and provide live/provider AQI information through an interactive dashboard.

## Models and algorithms
- Logistic Regression: linear baseline classifier.
- Decision Tree: rule-based tree classifier.
- Random Forest: ensemble of decision trees using bootstrap samples and feature subsets.
- Extra Trees: randomized tree ensemble.
- Gradient Boosting: sequential ensemble that improves residual errors.
- Dummy Classifier: simple baseline.
- Isolation Forest: unsupervised anomaly detection.
- SHAP: model explanation when compatible with the fitted estimator and package version.
- Permutation Importance: model-agnostic feature relevance diagnostic.

## Important metrics
- Accuracy: fraction of all predictions that are correct.
- Precision: fraction of predicted members of a class that are correct.
- Recall: fraction of actual members of a class that are found.
- Macro F1: average F1 across classes, weighting each class equally.
- Weighted F1: average F1 weighted by class support.
- Balanced Accuracy: average recall across classes.
- MCC: correlation between actual and predicted labels; useful with imbalanced classes.
- Cohen's Kappa: agreement adjusted for chance.

## Why macro F1?
AQI categories may be imbalanced. Macro F1 prevents majority classes from dominating the model-selection metric.

## What does Isolation Forest do?
It isolates observations through random feature splits. Unusual observations tend to require fewer splits to isolate and are marked as anomalies. This indicates statistical rarity, not necessarily a real-world emergency.

## Why SHAP?
SHAP estimates how each feature contributes to a model output relative to a background distribution. It explains model behavior; it does not prove pollutant causality.

## Why use permutation importance?
It measures how predictive performance changes when a feature is shuffled. A larger decrease indicates the model relies more heavily on that feature.

## What is the 3D chart?
It visualizes PM2.5, ozone and CO AQI sub-index values in three axes. It is not a geographical map.

## Critical limitations to explain honestly
- The CSV is cross-sectional and has no timestamp for training a temporal forecast model.
- It lacks trusted coordinates for precise historical hotspot mapping.
- The model predicts the existing AQI category; it does not predict future AQI.
- Open-Meteo's forecast page uses provider forecast values, not a trained LSTM.
- Health advice is generic and should not replace official public-health guidance.

## Likely examiner question: Why is accuracy so high?
The target category agrees strongly with PM2.5 AQI category in this dataset. This means the task may be strongly associated with one input feature and its labelling rules. We therefore inspect target dependence and compare the four-feature model with a PM2.5-only diagnostic. High accuracy must not be presented as proof of generalization to future dates or other monitoring systems.
