import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from src.simulator import generate_sensor_data
from src.features import SENSOR_COLUMNS, add_derived_features

st.set_page_config(page_title="SentinelEdge AI", page_icon="🛡️", layout="wide")
st.title("🛡️ SentinelEdge AI")
st.caption("AI-powered industrial sensor monitoring and anomaly detection")
st.markdown("Monitor simulated industrial telemetry, flag unusual operating conditions, and inspect sensor behavior through an interactive dashboard.")
MODEL_PATH = "models/anomaly_detector.joblib"
FEATURE_PATH = "models/feature_columns.joblib"
with st.sidebar:
    st.header("Simulation controls")
    n_samples = st.slider("Number of sensor readings", 100, 3000, 800, step=100)
    anomaly_rate = st.slider("Injected anomaly rate (%)", 1, 15, 5)
    random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    st.info("Demo mode uses generated telemetry. Adapt the input pipeline before connecting real sensors.")
raw = generate_sensor_data(n_samples=int(n_samples), anomaly_rate=float(anomaly_rate)/100, random_state=int(random_seed))
data = add_derived_features(raw)
feature_columns = ["temperature_c", "pressure_kpa", "vibration_mm_s", "power_kw", "temp_pressure_ratio", "vibration_power_ratio"]
if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
    model = joblib.load(MODEL_PATH)
    trained_features = joblib.load(FEATURE_PATH)
    X = data[trained_features]
    data["prediction"] = model.predict(X)
    data["anomaly_score"] = -model.decision_function(X)
    data["status"] = np.where(data["prediction"] == -1, "Anomaly", "Normal")
    source_note = "Loaded trained Isolation Forest model"
else:
    data["status"] = np.where(data["ground_truth_anomaly"], "Anomaly", "Normal")
    data["anomaly_score"] = np.where(data["ground_truth_anomaly"], 1.0, 0.0)
    source_note = "Simulation labels; run `python train_model.py` to enable model predictions"
st.caption(f"Analysis mode: {source_note}")
total = len(data); anomalies = int((data["status"] == "Anomaly").sum()); normal = total-anomalies
c1,c2,c3,c4=st.columns(4)
c1.metric("Readings analyzed", f"{total:,}"); c2.metric("Anomalies flagged", f"{anomalies:,}"); c3.metric("Normal readings", f"{normal:,}"); c4.metric("Flag rate", f"{(anomalies/total*100 if total else 0):.1f}%")
st.divider()
left,right=st.columns([1.5,1])
with left:
    st.subheader("Sensor telemetry")
    sensor_choice=st.selectbox("Choose a sensor", SENSOR_COLUMNS, index=2)
    fig=px.line(data,x="timestamp",y=sensor_choice,color="status",color_discrete_map={"Normal":"#16803d","Anomaly":"#d62728"},title=f"{sensor_choice.replace('_',' ').title()} over time")
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.subheader("Detection distribution")
    counts=data["status"].value_counts().rename_axis("status").reset_index(name="count")
    st.plotly_chart(px.pie(counts,names="status",values="count",hole=.55,color="status",color_discrete_map={"Normal":"#16803d","Anomaly":"#d62728"}),use_container_width=True)
st.subheader("Sensor relationships")
st.plotly_chart(px.scatter(data,x="vibration_mm_s",y="power_kw",color="status",color_discrete_map={"Normal":"#16803d","Anomaly":"#d62728"},hover_data=["timestamp","temperature_c","pressure_kpa","anomaly_score"],labels={"vibration_mm_s":"Vibration (mm/s)","power_kw":"Power (kW)"}),use_container_width=True)
st.subheader("Flagged readings")
show_cols=["timestamp"]+SENSOR_COLUMNS+["anomaly_score","status"]
flagged=data[data["status"]=="Anomaly"].sort_values("anomaly_score",ascending=False)
if flagged.empty: st.success("No anomalies were flagged in this sample.")
else: st.dataframe(flagged[show_cols].head(100),use_container_width=True,hide_index=True)
st.download_button("Download analysis as CSV",data=data[show_cols].to_csv(index=False).encode("utf-8"),file_name="sentineledge_analysis.csv",mime="text/csv")
with st.expander("About the model and limitations"):
    st.write("Isolation Forest identifies readings that differ from learned patterns; it does not diagnose a specific mechanical failure. This repository uses synthetic data for demonstration. Validate on representative site data before operational use.")
