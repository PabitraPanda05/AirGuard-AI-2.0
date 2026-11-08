import json
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

import requests

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# AirGuard AI — Streamlit frontend
# Expected model artifact:
# models/airguard_best_model.joblib
# Expected dataset:
# dataset/global_air_pollution_dataset.csv
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "airguard_best_model.joblib"
# Support both a normal project layout and the nested folder layout that can
# result when the ZIP is extracted into an existing project directory.
PROJECT_ROOTS = [ROOT, ROOT.parent, ROOT / "AirGuard_AI_Streamlit"]
DATA_PATHS = [
    base / "dataset" / "global_air_pollution_dataset.csv" for base in PROJECT_ROOTS
] + [base / "global_air_pollution_dataset.csv" for base in PROJECT_ROOTS]
MODEL_PATHS = [base / "models" / "airguard_best_model.joblib" for base in PROJECT_ROOTS]
METRICS_PATHS = [base / "models" / "model_metrics.json" for base in PROJECT_ROOTS]
RESULTS_PATHS = [base / "models" / "model_comparison.csv" for base in PROJECT_ROOTS]
MODEL_PATH = next((p for p in MODEL_PATHS if p.exists()), MODEL_PATHS[0])
METRICS_PATH = next((p for p in METRICS_PATHS if p.exists()), METRICS_PATHS[0])
RESULTS_PATH = next((p for p in RESULTS_PATHS if p.exists()), RESULTS_PATHS[0])

st.set_page_config(
    page_title="AirGuard AI | Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: rgba(255,255,255,0.075);
        --border: rgba(255,255,255,0.14);
        --text: #edf5ff;
        --muted: #9fb0c7;
        --cyan: #56e0d1;
        --blue: #7aa7ff;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 12% 8%, rgba(39, 190, 177, .18), transparent 28%),
          radial-gradient(circle at 88% 12%, rgba(80, 116, 255, .19), transparent 30%),
          linear-gradient(135deg, #07111f 0%, #0b1728 52%, #101a31 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,31,52,.96), rgba(8,20,36,.96));
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
    h1, h2, h3 { font-family: 'Outfit', sans-serif !important; color: #f4f8ff !important; }
    p, label, span, div { }
    .hero {
        padding: 30px 34px; border: 1px solid rgba(255,255,255,.18);
        border-radius: 26px; background: linear-gradient(120deg, rgba(67,215,197,.15), rgba(108,139,255,.13));
        box-shadow: 0 18px 60px rgba(0,0,0,.24); backdrop-filter: blur(18px);
        margin-bottom: 24px;
    }
    .eyebrow { color: #72e8d8; text-transform: uppercase; letter-spacing: 2.4px; font-size: .75rem; font-weight: 700; }
    .hero-title { font-family: 'Outfit'; font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.08; font-weight: 800; margin: 8px 0 12px; }
    .hero-sub { color: #b8c8dc; max-width: 780px; font-size: 1rem; }
    .glass-card { perspective: 1000px; transform-style: preserve-3d; transition: transform .25s ease, box-shadow .25s ease; 
        background: rgba(255,255,255,.075); border: 1px solid rgba(255,255,255,.14);
        border-radius: 20px; padding: 20px 22px; backdrop-filter: blur(16px);
        box-shadow: 0 12px 34px rgba(0,0,0,.16); min-height: 118px;
    }
    .glass-card:hover { transform: translateY(-4px) rotateX(1deg); box-shadow: 0 20px 48px rgba(0,0,0,.28); }\n    .metric-label { color: #9fb2ca; font-size: .82rem; font-weight: 600; letter-spacing: .4px; }
    .metric-value { color: #f5f9ff; font-family: 'Outfit'; font-size: 1.9rem; font-weight: 700; margin-top: 7px; }
    .metric-note { color: #78dfd2; font-size: .78rem; margin-top: 3px; }
    .section-note { color: #9fb0c7; font-size: .9rem; }
    .pill {
        display:inline-block; padding:7px 13px; border-radius:999px;
        background:rgba(86,224,209,.12); border:1px solid rgba(86,224,209,.28);
        color:#8cf0e2; font-size:.78rem; font-weight:700;
    }
    .result-box {
        padding: 24px; border-radius: 22px; border: 1px solid rgba(255,255,255,.2);
        background: linear-gradient(135deg, rgba(86,224,209,.16), rgba(122,167,255,.16));
        backdrop-filter: blur(18px); margin: 12px 0 20px;
    }
    div.stButton > button {
        border: 0; border-radius: 12px; padding: .65rem 1.2rem; font-weight: 700;
        color: #061522; background: linear-gradient(90deg, #56e0d1, #8cb8ff);
        box-shadow: 0 8px 24px rgba(86,224,209,.18);
    }
    div.stButton > button:hover { color:#061522; transform: translateY(-1px); }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 14px; }
    .small-muted { color: #9fb0c7; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]


@st.cache_resource
def load_artifact():
    path = next((p for p in MODEL_PATHS if p.exists()), None)
    if path is None:
        return None
    return joblib.load(path)


@st.cache_data
def load_dataset():
    for path in DATA_PATHS:
        if path.exists():
            data = pd.read_csv(path)
            data.columns = data.columns.astype(str).str.strip()
            return data
    return None


def glass_metric(label, value, note=""):
    st.markdown(
        f"""<div class="glass-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def hero(title, subtitle, tag="AIR QUALITY INTELLIGENCE"):
    st.markdown(
        f"""<div class="hero">
        <div class="eyebrow">{tag}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-sub">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def load_metrics():
    path = next((p for p in METRICS_PATHS if p.exists()), None)
    if path is not None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def category_color(category):
    name = str(category).lower()
    if "good" in name:
        return "#36d6a0"
    if "moderate" in name:
        return "#f4cf62"
    if "unhealthy" in name:
        return "#ff9b68" if "sensitive" in name else "#ff6f7d"
    if "hazard" in name:
        return "#c58cff"
    return "#8cb8ff"



# --------------------------- Live data / reporting helpers --------------------------
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
CITY_COORDS = {
    "Bhubaneswar": (20.2961, 85.8245), "Cuttack": (20.4625, 85.8830),
    "Puri": (19.8135, 85.8312), "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777), "Pune": (18.5204, 73.8567),
    "Kolkata": (22.5726, 88.3639), "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714), "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462), "Patna": (25.5941, 85.1376),
    "Ranchi": (23.3441, 85.3096), "Visakhapatnam": (17.6868, 83.2185),
    "New York": (40.7128, -74.0060), "London": (51.5072, -0.1276),
    "Paris": (48.8566, 2.3522), "Tokyo": (35.6762, 139.6503),
    "Singapore": (1.3521, 103.8198), "Beijing": (39.9042, 116.4074),
    "Los Angeles": (34.0522, -118.2437), "Dubai": (25.2048, 55.2708),
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_air_quality(latitude, longitude):
    params = {
        "latitude": latitude, "longitude": longitude,
        "current": "us_aqi,us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,us_aqi_ozone,us_aqi_carbon_monoxide,pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("current"):
        raise ValueError("The live provider did not return current air-quality values for this location.")
    return payload



@st.cache_data(ttl=900, show_spinner=False)
def fetch_air_quality_forecast(latitude, longitude):
    params = {"latitude": latitude, "longitude": longitude,
              "hourly": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,us_aqi_ozone,us_aqi_carbon_monoxide",
              "past_days": 1, "forecast_days": 3, "timezone": "auto"}
    response = requests.get(OPEN_METEO_AQ_URL, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("hourly", {}).get("time"):
        raise ValueError("No hourly AQI forecast data returned for this location.")
    return payload

def aqi_category(value):
    if pd.isna(value): return "Unavailable"
    value=float(value)
    if value <= 50: return "Good"
    if value <= 100: return "Moderate"
    if value <= 150: return "Unhealthy for Sensitive Groups"
    if value <= 200: return "Unhealthy"
    if value <= 300: return "Very Unhealthy"
    return "Hazardous"

def precaution_for_aqi(value):
    advice={"Good":"Air quality is satisfactory. Enjoy normal outdoor activities.",
    "Moderate":"Unusually sensitive people should consider reducing prolonged, intense outdoor exertion.",
    "Unhealthy for Sensitive Groups":"Children, older adults and people with heart or lung conditions should limit prolonged outdoor exertion.",
    "Unhealthy":"Everyone should reduce prolonged outdoor exertion; sensitive groups should avoid strenuous outdoor activity.",
    "Very Unhealthy":"Avoid strenuous outdoor activity; sensitive groups should remain indoors where possible.",
    "Hazardous":"Avoid outdoor activity and follow local public-health guidance."}
    return advice.get(aqi_category(value), "Air-quality guidance is unavailable.")

def send_aqi_email(recipient, city, value, threshold):
    import os, smtplib
    from email.message import EmailMessage
    try:
        cfg=st.secrets.get("email", {})
        host=cfg.get("smtp_host", os.getenv("AIRGUARD_SMTP_HOST", "smtp.gmail.com"))
        port=int(cfg.get("smtp_port", os.getenv("AIRGUARD_SMTP_PORT", "587")))
        sender=cfg.get("sender", os.getenv("AIRGUARD_SMTP_SENDER", ""))
        password=cfg.get("app_password", os.getenv("AIRGUARD_SMTP_APP_PASSWORD", ""))
    except Exception:
        host=os.getenv("AIRGUARD_SMTP_HOST", "smtp.gmail.com"); port=int(os.getenv("AIRGUARD_SMTP_PORT", "587"))
        sender=os.getenv("AIRGUARD_SMTP_SENDER", ""); password=os.getenv("AIRGUARD_SMTP_APP_PASSWORD", "")
    if not sender or not password:
        raise RuntimeError("Configure AIRGUARD_SMTP_SENDER and AIRGUARD_SMTP_APP_PASSWORD, or [email] in Streamlit secrets.")
    msg=EmailMessage(); msg["Subject"]=f"AirGuard AQI alert: {city} reached {value}"; msg["From"]=sender; msg["To"]=recipient
    msg.set_content(f"AirGuard AI pollution alert\nLocation: {city}\nCurrent US AQI: {value}\nThreshold: {threshold}\nCategory: {aqi_category(value)}\nPrecaution: {precaution_for_aqi(value)}\n\nSource: Open-Meteo modelled grid estimate, not an official station reading.")
    with smtplib.SMTP(host,port,timeout=20) as smtp:
        smtp.starttls(); smtp.login(sender,password); smtp.send_message(msg)

def live_model_features(current):
    # Open-Meteo provides pollutant-specific US AQI sub-indices, matching the
    # four input feature types used by the trained AirGuard classifier.
    return pd.DataFrame([{
        "CO AQI Value": current.get("us_aqi_carbon_monoxide", 0),
        "Ozone AQI Value": current.get("us_aqi_ozone", 0),
        "NO2 AQI Value": current.get("us_aqi_nitrogen_dioxide", 0),
        "PM2.5 AQI Value": current.get("us_aqi_pm2_5", 0),
    }], columns=FEATURES)

def make_prediction_pdf(title, details, feature_values=None, prediction=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export needs reportlab. Install it with: pip install reportlab") from exc
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], textColor=colors.HexColor("#526174"), leading=16))
    story = [Paragraph("AirGuard AI", styles["Title"]), Paragraph(title, styles["Heading2"]),
             Paragraph("Generated: " + datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z"), styles["Subtle"]), Spacer(1, 16)]
    rows = [["Field", "Value"]] + [[str(k), str(v)] for k, v in details.items()]
    if feature_values is not None:
        rows += [["Input feature", "AQI sub-index"]] + [[str(k), f"{float(v):.2f}"] for k, v in feature_values.items()]
    if prediction is not None:
        rows += [["Predicted AQI category", str(prediction)]]
    table = Table(rows, colWidths=[220, 250], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d5dee8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story += [table, Spacer(1, 18), Paragraph("Academic estimate only. This report is not an official monitoring record, regulatory assessment, or medical advice. Live values are provided by Open-Meteo and may differ from local monitoring stations.", styles["Subtle"])]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def explain_prediction(model, input_df):
    """Return SHAP values where supported; otherwise a clearly labelled local perturbation explanation."""
    try:
        import shap
        estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        if hasattr(model, "named_steps"):
            background_raw = df[FEATURES].dropna().sample(min(100, len(df[FEATURES].dropna())), random_state=42)
            background = model[:-1].transform(background_raw)
            explain_input = model[:-1].transform(input_df)
        else:
            background = df[FEATURES].dropna().sample(min(100, len(df[FEATURES].dropna())), random_state=42).to_numpy()
            explain_input = input_df.to_numpy()
        explainer = shap.Explainer(estimator, background, feature_names=FEATURES)
        values = explainer(explain_input)
        vals = np.asarray(values.values)
        if vals.ndim == 3:
            class_index = list(estimator.classes_).index(model.predict(input_df)[0])
            vals = vals[0, :, class_index]
        else:
            vals = vals.reshape(-1, len(FEATURES))[0]
        return pd.DataFrame({"Feature": FEATURES, "SHAP value": vals}), "SHAP explanation"
    except Exception:
        baseline = model.predict(input_df)[0]
        base_proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        rows = []
        for feature in FEATURES:
            changed = input_df.copy()
            changed.loc[:, feature] = 0
            if base_proba is not None:
                probs = model.predict_proba(changed)[0]
                idx = list(model.classes_).index(baseline)
                impact = float(base_proba[idx] - probs[idx])
            else:
                impact = float(model.predict(changed)[0] != baseline)
            rows.append((feature, impact))
        return pd.DataFrame(rows, columns=["Feature", "Local impact (not SHAP)"]), "Local perturbation explanation (SHAP unavailable)"


# --------------------------- Sidebar --------------------------
with st.sidebar:
    st.markdown("## 🌿 AirGuard AI")
    st.caption("Intelligent Air Quality Analysis")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "AQI Prediction", "Live AQI", "AQI Forecast", "AQI Map", "Pollution Hotspots", "Anomaly Detection", "3D Analytics", "Health Advisory", "AQI Alerts", "Explainable AI", "PDF Report", "Data Explorer", "City Comparison", "Model Insights", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<span class="pill">ML POWERED</span>', unsafe_allow_html=True)
    st.caption("Classification based on pollutant AQI sub-index values.")
    st.markdown("<div class='small-muted'>AirGuard AI · Academic project</div>", unsafe_allow_html=True)

artifact = load_artifact()
df = load_dataset()
metrics = load_metrics()

# --------------------------- Overview --------------------------
if page == "Overview":
    hero(
        "Breathe smarter. Understand your air.",
        "An interactive machine-learning workspace to explore air pollution patterns and classify air quality from pollutant AQI sub-index values.",
    )
    if df is None:
        st.warning("Dataset not found. Place global_air_pollution_dataset.csv inside the dataset/ folder.")
    else:
        total = len(df)
        cities = df["City"].nunique() if "City" in df.columns else "—"
        countries = df["Country"].nunique() if "Country" in df.columns else "—"
        avg_aqi = df["AQI Value"].mean() if "AQI Value" in df.columns else np.nan
        c1, c2, c3, c4 = st.columns(4)
        with c1: glass_metric("TOTAL RECORDS", f"{total:,}", "Dataset observations")
        with c2: glass_metric("CITIES COVERED", f"{cities:,}" if isinstance(cities, (int, np.integer)) else cities, "Unique cities")
        with c3: glass_metric("COUNTRIES", f"{countries:,}" if isinstance(countries, (int, np.integer)) else countries, "Geographical coverage")
        with c4: glass_metric("AVERAGE AQI", f"{avg_aqi:.1f}" if pd.notna(avg_aqi) else "—", "Dataset average")
        st.write("")
        left, right = st.columns([1.15, .85])
        with left:
            st.markdown("### AQI category distribution")
            if "AQI Category" in df.columns:
                counts = df["AQI Category"].value_counts().rename_axis("Category").reset_index(name="Records")
                fig = px.bar(counts, x="Category", y="Records", color="Category",
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             template="plotly_dark")
                fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("### Pollution snapshot")
            if "AQI Value" in df.columns:
                fig = px.histogram(df, x="AQI Value", nbins=35, template="plotly_dark",
                                   color_discrete_sequence=["#56e0d1"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
    st.markdown("### What you can do")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="glass-card"><h3>🔮 Predict</h3><p class="section-note">Enter four pollutant AQI sub-index values and classify the AQI category.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="glass-card"><h3>📊 Explore</h3><p class="section-note">Study pollutant distributions, category patterns and city-level records.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="glass-card"><h3>🧠 Evaluate</h3><p class="section-note">Review model comparison and held-out test metrics from the notebook.</p></div>', unsafe_allow_html=True)

# --------------------------- Prediction --------------------------
elif page == "AQI Prediction":
    hero("AQI category prediction", "Provide pollutant AQI sub-index values to receive a model-generated category estimate.", "PREDICTION STUDIO")
    if artifact is None:
        st.error("Trained model not found. Run all notebook cells through Step 12 to create models/airguard_best_model.joblib.")
        st.code("models/airguard_best_model.joblib", language="text")
    else:
        model = artifact["model"]
        st.markdown('<span class="pill">MODEL: ' + str(artifact.get("model_name", "Loaded model")) + '</span>', unsafe_allow_html=True)
        st.markdown("#### Enter pollutant AQI sub-index values")
        st.caption("These inputs are AQI sub-index values, not raw pollutant concentrations such as µg/m³ or ppm.")
        col1, col2 = st.columns(2)
        with col1:
            co = st.number_input("CO AQI Value", min_value=0.0, max_value=500.0, value=1.0, step=1.0,
                                 help="Carbon monoxide AQI sub-index")
            ozone = st.number_input("Ozone AQI Value", min_value=0.0, max_value=500.0, value=35.0, step=1.0,
                                     help="Ozone AQI sub-index")
        with col2:
            no2 = st.number_input("NO2 AQI Value", min_value=0.0, max_value=500.0, value=3.0, step=1.0,
                                  help="Nitrogen dioxide AQI sub-index")
            pm25 = st.number_input("PM2.5 AQI Value", min_value=0.0, max_value=500.0, value=65.0, step=1.0,
                                   help="Fine particulate matter AQI sub-index")
        if st.button("✨ Predict air quality", width="stretch"):
            input_df = pd.DataFrame([{
                "CO AQI Value": co,
                "Ozone AQI Value": ozone,
                "NO2 AQI Value": no2,
                "PM2.5 AQI Value": pm25,
            }])[artifact.get("features", FEATURES)]
            prediction = model.predict(input_df)[0]
            color = category_color(prediction)
            st.markdown(
                f"""<div class="result-box">
                <div class="eyebrow">PREDICTED AQI CATEGORY</div>
                <div style="font-family:Outfit;font-size:2.5rem;font-weight:800;color:{color};margin-top:8px">{prediction}</div>
                <div class="small-muted">This is a model classification based on the supplied AQI sub-index values.</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame({"AQI Category": model.classes_, "Probability": probs})
                prob_df = prob_df.sort_values("Probability", ascending=False)
                st.markdown("#### Class probability")
                fig = px.bar(prob_df, x="Probability", y="AQI Category", orientation="h",
                             color="Probability", color_continuous_scale=["#315a75", "#56e0d1"],
                             template="plotly_dark", range_x=[0, 1])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, width="stretch")
            st.session_state["last_prediction"] = {"features": input_df.iloc[0].to_dict(), "prediction": str(prediction), "source": "Manual prediction"}
            with st.expander("Why did the model make this prediction?", expanded=True):
                explanation, explanation_label = explain_prediction(model, input_df)
                st.caption(explanation_label)
                value_col = explanation.columns[1]
                explain_fig = px.bar(explanation.sort_values(value_col), x=value_col, y="Feature", orientation="h", template="plotly_dark")
                explain_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(explain_fig, width="stretch")
            report_details = {"Report type": "Manual model prediction", "Model": artifact.get("model_name", "AirGuard model")}
            st.download_button("📄 Download prediction PDF", make_prediction_pdf("AQI Prediction Report", report_details, input_df.iloc[0].to_dict(), prediction), "airguard_prediction_report.pdf", "application/pdf")
            st.info("For public-health or regulatory decisions, use validated monitoring equipment and the official AQI calculation method. This academic model is not an official alert system.")

# --------------------------- Live AQI --------------------------
elif page == "Live AQI":
    hero("Live air quality", "Check current air-quality conditions from Open-Meteo. No API key is required.", "LIVE MONITORING")
    city = st.selectbox("Choose a location", list(CITY_COORDS.keys()), index=list(CITY_COORDS).index("Bhubaneswar"))
    auto_refresh = st.toggle("Auto-refresh every 5 minutes", value=False)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=300000, key="airguard_live_auto_refresh")
        except ImportError:
            st.warning("Install streamlit-autorefresh to enable automatic refresh.")
    lat, lon = CITY_COORDS[city]
    if st.button("↻ Refresh live AQI", type="primary") or "live_payload" not in st.session_state or st.session_state.get("live_city") != city:
        try:
            with st.spinner("Fetching current air-quality data…"):
                st.session_state["live_payload"] = fetch_live_air_quality(lat, lon)
                st.session_state["live_city"] = city
        except Exception as exc:
            st.error(f"Could not retrieve live data: {exc}")
    payload = st.session_state.get("live_payload") if st.session_state.get("live_city") == city else None
    if payload:
        current = payload["current"]
        a, b, c = st.columns(3)
        with a: glass_metric("US AQI", current.get("us_aqi", "—"), "Open-Meteo current value")
        with b: glass_metric("PM2.5", f"{current.get('pm2_5', float('nan')):.1f} µg/m³" if current.get("pm2_5") is not None else "—", "Current concentration")
        with c: glass_metric("PM10", f"{current.get('pm10', float('nan')):.1f} µg/m³" if current.get("pm10") is not None else "—", "Current concentration")
        st.caption(f"Location: {city} ({lat:.4f}, {lon:.4f}) · Provider timestamp: {current.get('time', '—')} · Cached for up to 15 minutes")
        if current.get("us_aqi") is not None:
            st.markdown(f"**Current category: {aqi_category(current['us_aqi'])}**")
            st.info(precaution_for_aqi(current["us_aqi"]))
        st.markdown("### Pollutant readings")
        pollutants = {"Carbon monoxide": current.get("carbon_monoxide"), "Nitrogen dioxide": current.get("nitrogen_dioxide"), "Ozone": current.get("ozone"), "PM2.5": current.get("pm2_5"), "PM10": current.get("pm10")}
        st.dataframe(pd.DataFrame([{"Pollutant": k, "Concentration (µg/m³)": v} for k,v in pollutants.items()]), hide_index=True, width="stretch")
        if artifact is not None:
            live_features = live_model_features(current)
            if live_features.notna().all(axis=None):
                live_prediction = artifact["model"].predict(live_features)[0]
                st.markdown("### AirGuard model estimate from live pollutant sub-indices")
                st.success(f"Predicted category: **{live_prediction}**")
                st.caption("This is the trained academic classifier's estimate, not the official US AQI category. Open-Meteo live pollutant-specific US AQI fields are passed into the same four model features.")
                st.session_state["last_prediction"] = {"features": live_features.iloc[0].to_dict(), "prediction": str(live_prediction), "source": f"Live AQI · {city}"}
                st.download_button("📄 Download live AQI PDF", make_prediction_pdf("Live AQI Report", {"City":city,"Latitude":lat,"Longitude":lon,"US AQI":current.get("us_aqi"),"Provider timestamp":current.get("time"),"Model":artifact.get("model_name","AirGuard model")}, live_features.iloc[0].to_dict(), live_prediction), "airguard_live_aqi_report.pdf", "application/pdf")
    st.caption("Source: Open-Meteo Air Quality API. Live coverage, update frequency and pollutant availability vary by location. This is not a substitute for an official local AQI station.")

# --------------------------- AQI Forecast --------------------------
elif page == "AQI Forecast":
    hero("24 / 48 / 72-hour AQI forecast", "Explore hourly AQI outlooks with recent provider history and pollutant forecasts.", "SHORT-TERM FORECAST")
    city=st.selectbox("Forecast location",list(CITY_COORDS.keys()),index=list(CITY_COORDS).index("Bhubaneswar"),key="forecast_city")
    horizon=st.segmented_control("Forecast horizon",options=["24 hours","48 hours","72 hours"],default="24 hours")
    hours={"24 hours":24,"48 hours":48,"72 hours":72}.get(horizon,24)
    lat,lon=CITY_COORDS[city]
    if st.button("Generate AQI forecast",type="primary") or st.session_state.get("forecast_city_cached") != city:
        try:
            with st.spinner("Loading hourly AQI forecast…"):
                st.session_state["forecast_payload"]=fetch_air_quality_forecast(lat,lon)
                st.session_state["forecast_city_cached"]=city
        except Exception as exc: st.error(f"Forecast unavailable: {exc}")
    payload=st.session_state.get("forecast_payload") if st.session_state.get("forecast_city_cached")==city else None
    if payload:
        h=payload["hourly"]
        f=pd.DataFrame({"Time":pd.to_datetime(h["time"]),"AQI":h.get("us_aqi",[]),"PM2.5":h.get("pm2_5",[]),"PM10":h.get("pm10",[]),"CO":h.get("carbon_monoxide",[]),"NO2":h.get("nitrogen_dioxide",[]),"Ozone":h.get("ozone",[])})
        # Provider timestamps are local wall-clock times; locate the current hour by matching local time.
        now=pd.Timestamp.now().floor("h")
        future=f[f["Time"]>=now].head(hours).copy(); past=f[f["Time"]<now].tail(24).copy()
        if future.empty: future=f.tail(hours).copy()
        if not future.empty:
            avg,peak=future["AQI"].mean(),future["AQI"].max()
            c1,c2,c3=st.columns(3)
            with c1: glass_metric("FORECAST AVG AQI",f"{avg:.0f}",f"Next {len(future)} hours")
            with c2: glass_metric("FORECAST PEAK AQI",f"{peak:.0f}","Highest forecast hour")
            with c3: glass_metric("PEAK CATEGORY",aqi_category(peak),"US AQI bands")
            past["Series"]="Recent provider history"; future["Series"]="Provider forecast"
            plot=pd.concat([past,future],ignore_index=True)
            fig=px.line(plot,x="Time",y="AQI",color="Series",markers=True,template="plotly_dark",title=f"{city}: recent history vs next {len(future)} hours")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,width="stretch")
            future["Category"]=future["AQI"].apply(aqi_category)
            st.dataframe(future,hide_index=True,width="stretch")
            st.download_button("Download hourly forecast CSV",future.to_csv(index=False).encode(),f"airguard_{city.lower().replace(' ','_')}_{hours}h_forecast.csv","text/csv")
            st.markdown("### Precaution for forecast peak")
            st.warning(precaution_for_aqi(peak))
            if artifact is not None:
                st.markdown("### Existing AirGuard classifier outlook")
                st.caption("The saved classifier estimates a category from pollutant-specific AQI sub-indices; it does not generate the numerical forecast curve.")
                indices={"CO AQI Value":"us_aqi_carbon_monoxide","Ozone AQI Value":"us_aqi_ozone","NO2 AQI Value":"us_aqi_nitrogen_dioxide","PM2.5 AQI Value":"us_aqi_pm2_5"}
                predictions=[]
                times=h["time"]
                for _,row in future.iterrows():
                    key=row["Time"].strftime("%Y-%m-%dT%H:%M")
                    if key not in times: continue
                    ix=times.index(key); vals={feature:h.get(api,[None]*len(times))[ix] for feature,api in indices.items()}
                    if all(pd.notna(v) for v in vals.values()):
                        predictions.append({"Time":row["Time"],"Provider AQI":row["AQI"],"AirGuard predicted category":str(artifact["model"].predict(pd.DataFrame([vals],columns=FEATURES))[0])})
                if predictions: st.dataframe(pd.DataFrame(predictions),hide_index=True,width="stretch")
        else: st.warning("No forecast rows were returned for this horizon.")
    st.caption("Forecast numbers come from Open-Meteo, not a newly trained AirGuard time-series model. The supplied global pollution dataset is cross-sectional and has no timestamps, so Random Forest, XGBoost or LSTM time-series models cannot be trained and validated honestly without historical hourly observations. The recent history line is provider data, not ground-truth station measurements.")

# --------------------------- AQI Map --------------------------
elif page == "AQI Map":
    hero("Air quality map", "Explore current US AQI readings for selected cities using verified city coordinates.", "GEOSPATIAL VIEW")
    chosen = st.multiselect("Cities to display", list(CITY_COORDS.keys()), default=["Bhubaneswar","Delhi","Mumbai","Kolkata","Chennai","Bengaluru"])
    map_rows=[]
    if chosen:
        progress=st.progress(0, text="Loading city readings…")
        for i, name in enumerate(chosen):
            la,lo=CITY_COORDS[name]
            try:
                curr=fetch_live_air_quality(la,lo)["current"]
                map_rows.append({"City":name,"Latitude":la,"Longitude":lo,"US AQI":curr.get("us_aqi"),"Category":aqi_category(curr.get("us_aqi")),"PM2.5":curr.get("pm2_5")})
            except Exception:
                map_rows.append({"City":name,"Latitude":la,"Longitude":lo,"US AQI":None,"PM2.5":None})
            progress.progress((i+1)/len(chosen), text=f"Checked {name}")
        progress.empty()
    map_df=pd.DataFrame(map_rows).dropna(subset=["US AQI"])
    if not map_df.empty:
        fig=px.scatter_map(map_df,lat="Latitude",lon="Longitude",color="Category",size="US AQI",hover_name="City",hover_data=["US AQI","PM2.5"],color_discrete_map={"Good":"#2ecc71","Moderate":"#f1c40f","Unhealthy for Sensitive Groups":"#e67e22","Unhealthy":"#e74c3c","Very Unhealthy":"#8e44ad","Hazardous":"#7f1d1d"},zoom=1.1,height=560,map_style="carto-positron")
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig,width="stretch")
        st.dataframe(map_df,hide_index=True,width="stretch")
        st.download_button("⬇ Download map readings",map_df.to_csv(index=False).encode("utf-8"),"airguard_city_live_aqi.csv","text/csv")
    else:
        st.warning("No live city readings were returned. Try again later or choose fewer cities.")
    st.caption("Coordinates are curated city-centre reference points. AQI values are modelled grid estimates from Open-Meteo, not measurements at the exact map pin.")

# --------------------------- AQI Alerts --------------------------
elif page == "AQI Alerts":
    hero("Personal AQI alerts", "Set a threshold and check whether the latest live AQI for a city crosses it.", "CUSTOM ALERTS")
    alert_city=st.selectbox("Alert location",list(CITY_COORDS.keys()),index=list(CITY_COORDS).index("Bhubaneswar"))
    threshold=st.slider("Alert when US AQI reaches or exceeds",0,300,100,5)
    recipient=st.text_input("Email recipient (optional)",placeholder="you@example.com")
    st.session_state["aqi_alert_config"]={"city":alert_city,"threshold":threshold}
    if st.button("Check alert now",type="primary"):
        la,lo=CITY_COORDS[alert_city]
        try:
            current=fetch_live_air_quality(la,lo)["current"]
            value=current.get("us_aqi")
            if value is None: st.warning("AQI value is unavailable for this location right now.")
            elif value>=threshold:
                st.error(f"⚠️ Alert: {alert_city} US AQI is {value}, at or above your threshold of {threshold}.")
                st.warning(precaution_for_aqi(value))
                if recipient:
                    try: send_aqi_email(recipient,alert_city,value,threshold); st.success(f"Email alert sent to {recipient}.")
                    except Exception as mail_exc: st.error(f"Email could not be sent: {mail_exc}")
            else:
                st.success(f"No threshold alert: {alert_city} US AQI is {value}, below {threshold}.")
                st.info(precaution_for_aqi(value))
            st.caption(f"Provider timestamp: {current.get('time','—')}. Checks run on demand; background/push alerts are not configured.")
        except Exception as exc: st.error(f"Unable to check live AQI: {exc}")
    st.info("Email alerts require SMTP credentials in environment variables or Streamlit secrets. Checks are user-initiated; background, SMS and browser push notifications are not configured.")

# --------------------------- Explainable AI --------------------------
elif page == "Explainable AI":
    hero("Explainable AI", "Understand which pollutant AQI sub-index values influenced a model prediction.", "MODEL TRANSPARENCY")
    if artifact is None:
        st.error("The trained model artifact is missing.")
    else:
        model=artifact["model"]
        if "last_prediction" in st.session_state:
            values=st.session_state["last_prediction"]["features"]
            st.caption(f"Explaining the latest prediction · {st.session_state['last_prediction'].get('source','Prediction')}")
        else:
            st.caption("Enter feature values below to create an explanation.")
            defaults=[1.0,35.0,3.0,65.0]
            values={}
            cols=st.columns(2)
            for i,feature in enumerate(FEATURES):
                with cols[i%2]: values[feature]=st.number_input(feature,0.0,500.0,defaults[i],1.0,key=f"explain_{i}")
        input_df=pd.DataFrame([values],columns=FEATURES)
        prediction=model.predict(input_df)[0]
        st.markdown(f"### Model prediction: {prediction}")
        explanation,label=explain_prediction(model,input_df)
        st.caption(label)
        valcol=explanation.columns[1]
        fig=px.bar(explanation.sort_values(valcol),x=valcol,y="Feature",orientation="h",template="plotly_dark",title="Feature contribution / local impact")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch")
        st.dataframe(input_df,width="stretch")
        st.caption("When SHAP can explain the selected estimator, SHAP values are shown. Otherwise the app shows a local perturbation comparison and labels it clearly; that fallback is not SHAP and should not be interpreted as a causal effect.")

# --------------------------- PDF Report --------------------------
elif page == "PDF Report":
    hero("Downloadable PDF report", "Generate a concise report from the latest prediction or from the saved model evaluation.", "REPORT CENTRE")
    last=st.session_state.get("last_prediction")
    if last:
        st.markdown(f"**Latest prediction:** {last['prediction']}  ")
        st.caption(f"Source: {last.get('source','Prediction')}")
        st.dataframe(pd.DataFrame([last["features"]]),hide_index=True,width="stretch")
        pdf=make_prediction_pdf("AirGuard Prediction Summary",{"Source":last.get("source","Prediction"),"Predicted category":last["prediction"],"Model":artifact.get("model_name","AirGuard model") if artifact else "Unavailable"},last["features"],last["prediction"])
        st.download_button("⬇ Download latest prediction report",pdf,"airguard_report.pdf","application/pdf",type="primary")
    else:
        st.info("Make a prediction in AQI Prediction or Live AQI first. You can still download the model evaluation summary below.")
    if metrics:
        summary={k:v for k,v in metrics.items() if isinstance(v,(str,int,float))}
        eval_pdf=make_prediction_pdf("Model Evaluation Summary",summary)
        st.download_button("⬇ Download model insights PDF",eval_pdf,"airguard_model_insights.pdf","application/pdf")

# --------------------------- Data Explorer --------------------------
elif page == "Data Explorer":
    hero("Explore the pollution dataset", "Filter, inspect and visualize the records used during model development.", "DATA INTELLIGENCE")
    if df is None:
        st.warning("Dataset not found. Add the CSV to the dataset/ folder.")
    else:
        filtered = df.copy()
        f1, f2 = st.columns(2)
        if "Country" in filtered.columns:
            countries = sorted(filtered["Country"].dropna().astype(str).unique())
            with f1:
                selected_countries = st.multiselect("Filter by country", countries, default=[])
            if selected_countries:
                filtered = filtered[filtered["Country"].astype(str).isin(selected_countries)]
        if "AQI Category" in filtered.columns:
            categories = sorted(filtered["AQI Category"].dropna().astype(str).unique())
            with f2:
                selected_categories = st.multiselect("Filter by AQI category", categories, default=[])
            if selected_categories:
                filtered = filtered[filtered["AQI Category"].astype(str).isin(selected_categories)]
        st.markdown(f"**{len(filtered):,} records** match the selected filters.")
        st.dataframe(filtered.head(1000), width="stretch", height=350)
        if "AQI Category" in filtered.columns:
            cat = filtered["AQI Category"].value_counts().reset_index()
            cat.columns = ["AQI Category", "Records"]
            fig = px.pie(cat, names="AQI Category", values="Records", hole=.55,
                         template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        pollutant = st.selectbox("Select pollutant feature", [x for x in FEATURES if x in filtered.columns])
        if pollutant:
            fig = px.histogram(filtered, x=pollutant, color="AQI Category" if "AQI Category" in filtered.columns else None,
                               nbins=40, template="plotly_dark", opacity=.8)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        st.download_button("⬇ Download filtered data", filtered.to_csv(index=False).encode("utf-8"),
                           "airguard_filtered_data.csv", "text/csv")

# --------------------------- City Comparison --------------------------
elif page == "City Comparison":
    hero("City pollution comparison", "Compare average AQI values across cities represented in the supplied dataset.", "GEOGRAPHICAL INSIGHTS")
    if df is None or not {"City", "AQI Value"}.issubset(df.columns):
        st.warning("City and AQI Value columns are required for this view.")
    else:
        # Normalize city names and AQI values before grouping. The supplied
        # dataset has approximately one observation per city, so default to 1.
        city_data = df[["City", "AQI Value"]].copy()
        city_data["City"] = city_data["City"].astype("string").str.strip()
        city_data["AQI Value"] = pd.to_numeric(city_data["AQI Value"], errors="coerce")
        city_data = city_data.dropna(subset=["City", "AQI Value"])
        city_data = city_data[city_data["City"].ne("")]

        city_stats = (
            city_data.groupby("City", as_index=False)
            .agg(
                Records=("AQI Value", "count"),
                Average_AQI=("AQI Value", "mean"),
                Maximum_AQI=("AQI Value", "max"),
            )
        )

        if city_stats.empty:
            st.warning("No usable city/AQI records were found. Check the City and AQI Value columns.")
        else:
            st.caption(
                f"Loaded {len(city_data):,} valid observations across "
                f"{city_stats['City'].nunique():,} cities. "
                "Most cities have one observation, so their average and maximum are the same."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                glass_metric("CITIES AVAILABLE", f"{len(city_stats):,}", "Cities with valid AQI")
            with c2:
                glass_metric("VALID OBSERVATIONS", f"{len(city_data):,}", "Rows included")
            with c3:
                glass_metric("OVERALL MEAN AQI", f"{city_data['AQI Value'].mean():.1f}", "Across available records")

            # This dataset has exactly one observation per city, so a
            # minimum-records slider is unnecessary. Include every city.
            eligible = city_stats.copy()

            st.info(
                "All cities have one observation in this dataset. "
                "All available cities are included in the comparison."
            )

            top_n = st.slider("Number of cities to display", 5, 40, 15, key="city_top_n_final")
            sort_label = st.radio(
                "Rank cities by", ["Average AQI", "Maximum AQI"], horizontal=True,
                key="city_sort_by_final",
            )
            sort_by = "Average_AQI" if sort_label == "Average AQI" else "Maximum_AQI"

            if eligible.empty:
                st.warning("No cities are available for comparison.")
            else:
                ranked_all = eligible.sort_values(sort_by, ascending=False)
                ranked = ranked_all.head(top_n).sort_values(sort_by, ascending=True)
                fig = px.bar(
                    ranked, x=sort_by, y="City", orientation="h",
                    color=sort_by,
                    color_continuous_scale=["#56e0d1", "#7aa7ff"],
                    template="plotly_dark",
                    hover_data={"Records": True, "Average_AQI": ":.1f", "Maximum_AQI": True},
                    labels={"Average_AQI": "Average AQI", "Maximum_AQI": "Maximum AQI"},
                    title=f"Top {min(top_n, len(ranked_all))} cities by {sort_label.lower()}",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, height=max(450, len(ranked) * 30),
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig, width="stretch")
                st.markdown("### City ranking")
                display_stats = ranked_all.rename(columns={
                    "Average_AQI": "Average AQI", "Maximum_AQI": "Maximum AQI"
                })
                st.dataframe(
                    display_stats[["City", "Records", "Average AQI", "Maximum AQI"]],
                    width="stretch", hide_index=True,
                )
                st.download_button(
                    "⬇ Download city comparison", display_stats.to_csv(index=False).encode("utf-8"),
                    "airguard_city_comparison.csv", "text/csv",
                )
                st.info(
                    "Interpretation note: this dataset contains nearly one row per city. "
                    "Treat the ranking as a comparison of the supplied observations, not as a "
                    "long-term city pollution average or trend."
                )

# --------------------------- Model Insights --------------------------
elif page == "Model Insights":
    hero("Model performance & insights", "Review the evaluation metrics exported by the Jupyter Notebook.", "MODEL LAB")
    if metrics:
        model_name = metrics.get("model_name", "Best model")
        st.markdown(f'<span class="pill">SELECTED MODEL: {model_name}</span>', unsafe_allow_html=True)
        metric_labels = [
            ("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"),
            ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")
        ]
        cols = st.columns(4)
        for col, (label, key) in zip(cols, metric_labels):
            value = metrics.get(key)
            with col:
                glass_metric(label.upper(), f"{value:.3f}" if isinstance(value, (int, float)) else "—",
                             "Held-out test set")
    else:
        st.info("Model metrics file not found. Run notebook Step 12 to export models/model_metrics.json.")
    results_file = next((p for p in RESULTS_PATHS if p.exists()), None)
    if results_file is not None:
        results = pd.read_csv(results_file)
        st.markdown("### Model comparison")
        st.dataframe(results, width="stretch")
        if "CV Macro F1 Mean" in results.columns:
            ordered = results.sort_values("CV Macro F1 Mean")
            fig = px.bar(ordered, x="CV Macro F1 Mean", y="Model", orientation="h",
                         error_x="CV Macro F1 Std" if "CV Macro F1 Std" in ordered.columns else None,
                         color="CV Macro F1 Mean", color_continuous_scale=["#315a75", "#56e0d1"],
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Model comparison CSV is not available yet.")
    st.warning("Accuracy should not be interpreted alone. Review macro F1 and per-class recall because AQI categories may be imbalanced.")

# --------------------------- About --------------------------

# --------------------------- Pollution Hotspots --------------------------
elif page == "Pollution Hotspots":
    hero("Pollution hotspot intelligence", "Rank locations in the supplied historical dataset by observed AQI. This is a city-level historical ranking, not a live sensor map.", "SPATIAL INTELLIGENCE")
    if df is None or not {"City", "AQI Value"}.issubset(df.columns):
        st.warning("City and AQI Value columns are required for this analysis.")
    else:
        top_n = st.slider("Number of cities to display", 5, min(30, max(5, df["City"].nunique())), min(15, max(5, df["City"].nunique())))
        city_stats = (df.groupby("City", dropna=True)["AQI Value"]
                      .agg(Mean_AQI="mean", Max_AQI="max", Records="count")
                      .sort_values("Mean_AQI", ascending=False).reset_index())
        c1, c2, c3 = st.columns(3)
        with c1: glass_metric("CITIES ANALYSED", f"{city_stats['City'].nunique():,}", "Historical dataset")
        with c2: glass_metric("HIGHEST CITY MEAN", f"{city_stats['Mean_AQI'].max():.1f}", "AQI value")
        with c3: glass_metric("RECORDS ANALYSED", f"{int(city_stats['Records'].sum()):,}", "Dataset rows")
        fig = px.bar(city_stats.head(top_n).sort_values("Mean_AQI"), x="Mean_AQI", y="City",
                     orientation="h", color="Mean_AQI", color_continuous_scale=["#56e0d1", "#ff9b68"],
                     hover_data=["Max_AQI", "Records"], template="plotly_dark",
                     title="Cities ranked by mean historical AQI")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(city_stats, use_container_width=True)
        st.caption("The source dataset contains city names but no trusted coordinates or timestamps. Do not interpret this ranking as current city conditions.")

# --------------------------- Anomaly Detection --------------------------
elif page == "Anomaly Detection":
    hero("Pollution anomaly detection", "Isolation Forest identifies unusual pollutant AQI combinations in the historical dataset.", "UNSUPERVISED INTELLIGENCE")
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if df is None or not all(f in df.columns for f in FEATURES):
        st.warning("The four pollutant AQI sub-index columns are required.")
    else:
        contamination = st.slider("Expected anomaly proportion", 0.01, 0.15, 0.03, 0.01)
        @st.cache_data(show_spinner=False)
        def run_anomaly_detection(frame, feature_names, contamination_value):
            values = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce")
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("detector", IsolationForest(n_estimators=250, contamination=contamination_value,
                                             random_state=42, n_jobs=-1))
            ])
            pipe.fit(values)
            labels = pipe.predict(values)
            scores = pipe.decision_function(values)
            out = frame.copy()
            out["Anomaly Status"] = np.where(labels == -1, "Anomaly", "Normal")
            out["Anomaly Score"] = scores
            return out, pipe
        anomaly_df, _ = run_anomaly_detection(df, tuple(FEATURES), contamination)
        flagged = anomaly_df[anomaly_df["Anomaly Status"] == "Anomaly"]
        a,b,c = st.columns(3)
        with a: glass_metric("FLAGGED RECORDS", f"{len(flagged):,}", "Statistical outliers")
        with b: glass_metric("ANOMALY RATE", f"{len(flagged)/max(len(anomaly_df),1):.1%}", "Selected contamination")
        with c: glass_metric("NORMAL RECORDS", f"{len(anomaly_df)-len(flagged):,}", "Model inliers")
        fig = px.scatter(anomaly_df.sample(min(5000, len(anomaly_df)), random_state=42),
                         x="PM2.5 AQI Value", y="Ozone AQI Value", color="Anomaly Status",
                         size="CO AQI Value", hover_data=["City"] if "City" in anomaly_df.columns else None,
                         color_discrete_map={"Normal":"#56e0d1","Anomaly":"#ff6f7d"},
                         template="plotly_dark", title="Pollutant pattern anomalies")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(flagged.head(500), use_container_width=True)
        csv = flagged.to_csv(index=False).encode("utf-8")
        st.download_button("Download flagged records CSV", csv, "airguard_anomalies.csv", "text/csv")
        st.warning("An anomaly is a statistical outlier, not proof of a hazardous event. Verify with official monitoring data.")

# --------------------------- 3D Analytics --------------------------
elif page == "3D Analytics":
    hero("3D pollutant analytics", "Explore relationships among PM2.5, ozone and CO AQI sub-indices in an interactive 3D space.", "VISUAL ANALYTICS")
    if df is None or not all(f in df.columns for f in FEATURES):
        st.warning("Pollutant AQI sub-index columns are not available.")
    else:
        sample_n = st.slider("Points to render", 500, min(10000, len(df)), min(4000, len(df)), 500)
        sample = df.sample(sample_n, random_state=42)
        color_by = st.selectbox(
    "Color points by",
    ["AQI Category", "City"] if "City" in df.columns else ["AQI Category"]
)
        fig = px.scatter_3d(sample, x="PM2.5 AQI Value", y="Ozone AQI Value", z="CO AQI Value",
                            color=color_by, hover_data=["City"] if "City" in sample.columns else None,
                            opacity=0.68, template="plotly_dark",
                            title="Interactive 3D pollutant relationship")
        fig.update_layout(height=720, paper_bgcolor="rgba(0,0,0,0)",
                          scene=dict(bgcolor="rgba(0,0,0,0)",
                                     xaxis_title="PM2.5 AQI sub-index",
                                     yaxis_title="Ozone AQI sub-index",
                                     zaxis_title="CO AQI sub-index"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rotate, zoom and hover to inspect the observations. This 3D chart visualizes pollutant sub-indices, not latitude/longitude.")

# --------------------------- Health Advisory --------------------------
elif page == "Health Advisory":
    hero("Health risk advisory", "View general precautionary guidance based on the latest provider AQI reading or a manually entered AQI value.", "PUBLIC AWARENESS")
    advisory_mode = st.radio("Advisory source", ["Fetch live AQI", "Enter AQI manually"], horizontal=True)
    if advisory_mode == "Fetch live AQI":
        city = st.selectbox("Select city", list(CITY_COORDS.keys()), index=list(CITY_COORDS).index("Bhubaneswar"), key="health_city")
        lat, lon = CITY_COORDS[city]
        if st.button("Fetch current AQI for advisory"):
            try:
                current = fetch_live_air_quality(lat, lon)["current"]
                val = current.get("us_aqi")
                if val is None:
                    st.warning("AQI value is unavailable from the provider.")
                else:
                    cat = aqi_category(val)
                    st.markdown(f'<div class="result-box"><div class="eyebrow">{city} · PROVIDER AQI</div><div class="hero-title">{float(val):.0f}</div><h3>{cat}</h3><p>{precaution_for_aqi(val)}</p></div>', unsafe_allow_html=True)
                    st.caption("Provider: Open-Meteo. Modelled grid estimate; not a calibrated local station measurement.")
            except Exception as exc:
                st.error(f"Could not retrieve live AQI: {exc}")
    else:
        manual_value = st.number_input("Enter AQI value", min_value=0, max_value=500, value=75, step=1)
        cat = aqi_category(manual_value)
        st.markdown(f'<div class="result-box"><div class="eyebrow">AQI CATEGORY</div><div class="hero-title">{manual_value}</div><h3>{cat}</h3><p>{precaution_for_aqi(manual_value)}</p></div>', unsafe_allow_html=True)
    st.info("This is general educational guidance, not personal medical advice. Follow local public-health and official AQI advisories.")

elif page == "About":
    hero("About AirGuard AI", "An air-quality intelligence prototype combining supervised ML, anomaly screening, live provider data, forecasting views, explainability and interactive analytics.", "PROJECT OVERVIEW")
    st.markdown("""
    <div class="glass-card">
      <h3>Project objective</h3>
      <p class="section-note">Analyze historical air pollution observations and train supervised machine-learning classifiers to estimate the AQI Category from four pollutant AQI sub-index values.</p>
      <h3>Input features</h3>
      <p class="section-note">CO AQI Value · Ozone AQI Value · NO2 AQI Value · PM2.5 AQI Value</p>
      <h3>Technology</h3>
      <p class="section-note">Python · Pandas · NumPy · Scikit-learn · Isolation Forest · SHAP · ReportLab · Open-Meteo · Plotly · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Important scope and limitations")
    st.markdown("""
    - The ML classifier predicts the dataset's existing **AQI Category**. The separate AQI Forecast page displays provider forecast data; it is not an LSTM model trained on the supplied CSV.
    - The model uses AQI sub-index values, not raw sensor measurements.
    - The dataset labels and model outputs are for academic exploration, not official environmental or medical guidance.
    - Live AQI is sourced from Open-Meteo and represents modelled grid estimates; it is not a calibrated local sensor reading.
    - AQI alerts are checked on demand and do not send background push notifications.
    """)
    st.markdown("### Expected project files")
    st.code(
        "AirGuard_AI_Streamlit/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "├── dataset/\n"
        "│   └── global_air_pollution_dataset.csv\n"
        "└── models/\n"
        "    ├── airguard_best_model.joblib\n"
        "    ├── model_metrics.json\n"
        "    └── model_comparison.csv",
        language="text",
    )

st.markdown(
    "<div style='text-align:center;color:#71849d;padding:30px 0 5px;font-size:.78rem'>"
    "AIRGUARD AI · Built with Python & Streamlit · Academic prototype"
    "</div>",
    unsafe_allow_html=True,
)
