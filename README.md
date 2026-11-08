# AirGuard AI 2.0 — Complete Project

An upgraded academic air-quality intelligence prototype based on the supplied AirGuard AI project.

## What is included

- `notebooks/AirGuard_AI_2_0_Complete.ipynb` — original end-to-end ML workflow plus advanced audit, comprehensive metrics, permutation importance, Isolation Forest anomaly detection, optional SHAP explainability and 3D exploratory analytics.
- `app.py` — Streamlit application with glassmorphism / 3D-inspired UI.
- `dataset/` — supplied air-pollution dataset.
- `models/` — existing selected classifier, comparison table and exported metrics.
- `outputs/` — location for generated notebook results.
- `requirements.txt` — Python dependencies.
- `run_app.bat` and `run_notebook.bat` — Windows launch helpers.

## Dashboard modules

1. Overview
2. AQI Prediction
3. Live AQI (Open-Meteo provider)
4. AQI Forecast (provider forecast view; not an LSTM trained on the static CSV)
5. AQI Map (provider current conditions for supported cities)
6. Pollution Hotspots (historical city ranking from supplied dataset)
7. Anomaly Detection (Isolation Forest)
8. 3D Analytics (interactive Plotly 3D pollutant chart)
9. Health Advisory (general AQI-category precautions)
10. AQI Alerts
11. Explainable AI
12. PDF Report
13. Data Explorer
14. City Comparison
15. Model Insights
16. About

## Setup (Windows / Anaconda)

Use the same Python environment in which your original AirGuard AI model was trained.

```bash
cd path\to\AirGuard_AI_2_0
python -m pip install -r requirements.txt
```

The supplied serialized classifier records scikit-learn 1.9.1, so the requirements pin is aligned to that version. Keep this version consistent when loading the saved Joblib model; cross-version loading can produce warnings or unpredictable behavior.

## Run the notebook

From the project root:

```bash
python -m notebook notebooks/AirGuard_AI_2_0_Complete.ipynb
```

Run all cells from the top in order. The original workflow trains and exports the classifier. The AirGuard AI 2.0 chapter then generates the audit, detailed metrics, permutation importance, anomaly detector, optional SHAP outputs and 3D visualization.

## Run the Streamlit dashboard

```bash
python -m streamlit run app.py
```

Or double-click `run_app.bat`.

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Live-data source

Live and forecast views call the Open-Meteo Air Quality API. Internet access is required. Provider values are modelled grid estimates and may differ from regulatory monitoring-station observations. If the API is unavailable, the live pages show an error rather than silently substituting fabricated readings.

## Important scientific limitations

- The supplied dataset is cross-sectional and does not contain an observation timestamp suitable for time-series model training.
- The supplied dataset does not contain trusted latitude/longitude columns. Historical hotspot analysis is therefore a city ranking, not a coordinate-accurate heat map.
- The supervised model classifies the existing dataset's AQI category from four pollutant AQI sub-index values. It is not a future AQI forecast.
- The app's forecast page displays provider forecast data. It must not be described as an LSTM trained on the supplied dataset.
- Isolation Forest flags unusual pollutant-feature combinations. Anomaly flags are not proof of a hazardous event.
- Health advisory text is general educational guidance, not a diagnosis or personal medical advice.
- Live provider data is not an official local station reading. For official decisions, consult the relevant pollution-control authority.

## Day 1 evaluation exports

After running the notebook, inspect `outputs/airguard_ai_2_0/` for:
- `target_dependence_audit.csv`
- `class_wise_metrics.csv`
- `evaluation_metrics.json`
- `confusion_matrix.png`
- `permutation_importance.csv` and `.png`
- `anomaly_detection_results.csv`
- `shap_importance.csv` and `.png` (only if SHAP succeeds)
- `airguard_ai_2_0_summary.json`
