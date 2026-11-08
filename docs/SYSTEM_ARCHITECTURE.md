# System Architecture — AirGuard AI 2.0

```text
                    AIRGUARD AI 2.0
                           |
            +--------------+--------------+
            |                             |
     Supplied CSV Dataset           Open-Meteo API
            |                       (live / forecast)
            |                             |
     Data Validation                 Provider Data
            |                             |
    Feature Engineering                 AQI Views
            |
    Supervised ML Classifier
            |
    +-------+----------+------------------+
    |                  |                  |
 Model Evaluation  Explainability    Prediction UI
    |                  |
 Confusion Matrix   SHAP / Permutation
 Macro F1, MCC      Importance
    |
    +-------------------------------+
                                    |
                       Unsupervised Anomaly Detection
                                    |
                          Isolation Forest
                                    |
                     Streamlit Intelligence Dashboard
                                    |
          +-------------+-----------+------------+
          |             |           |            |
       3D Analytics  City ranking  Advisory   PDF report
```

## Data flow

1. The notebook loads and validates the supplied CSV.
2. Four pollutant AQI sub-index values are used as supervised model inputs.
3. The selected classifier predicts the existing AQI category.
4. Evaluation uses a held-out test set and class-sensitive metrics.
5. Permutation importance and optional SHAP provide model explanation.
6. Isolation Forest identifies statistical outliers in the four-feature space.
7. The Streamlit app loads the saved model and exposes prediction, analytics and reporting pages.
8. Live and forecast pages retrieve current/provider forecast data independently through Open-Meteo.

## Design decision

The static dataset and live provider feed are kept as separate data sources. They are not merged into a fabricated training history. This prevents confusion between retrospective classification and future AQI forecasting.
