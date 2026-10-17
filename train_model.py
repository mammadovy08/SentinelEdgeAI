import os
import joblib
from sklearn.ensemble import IsolationForest
from src.simulator import generate_sensor_data
from src.features import add_derived_features
FEATURE_COLUMNS=["temperature_c","pressure_kpa","vibration_mm_s","power_kw","temp_pressure_ratio","vibration_power_ratio"]
def main():
    os.makedirs("models",exist_ok=True)
    training=generate_sensor_data(n_samples=6000,anomaly_rate=0.01,random_state=2026)
    X=add_derived_features(training)[FEATURE_COLUMNS]
    model=IsolationForest(n_estimators=250,contamination=0.03,random_state=2026,n_jobs=-1)
    model.fit(X)
    joblib.dump(model,"models/anomaly_detector.joblib")
    joblib.dump(FEATURE_COLUMNS,"models/feature_columns.joblib")
    print("Model saved to models/anomaly_detector.joblib")
    print("Demo model trained on synthetic data; validate before real-world use.")
if __name__=="__main__": main()
