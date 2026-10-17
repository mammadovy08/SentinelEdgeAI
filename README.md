# SentinelEdge AI
### Intelligent Industrial Sensor Monitoring & Anomaly Detection

SentinelEdge AI is a portfolio demonstration of an AI-assisted industrial monitoring workflow. It generates telemetry for four sensor channels, derives additional features, detects unusual readings with an Isolation Forest model, and displays results in an interactive Streamlit dashboard.

**Stack:** Python · scikit-learn · Streamlit · Plotly · Pandas · NumPy

## Features
- Simulates temperature, pressure, vibration, and power telemetry.
- Injects configurable abnormal readings for repeatable demos.
- Trains an unsupervised Isolation Forest detector.
- Visualizes trends, anomaly distribution, and sensor relationships.
- Lists flagged readings and exports results as CSV.
- Includes automated tests.

## Quick start
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Run tests
```bash
pytest -q
```

## Structure
```text
SentinelEdgeAI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── simulator.py
└── tests/
    └── test_simulator.py
```

The `models/` directory is created by the training script and intentionally excluded from Git. Clone the repository and run `python train_model.py` to recreate the model locally.

## How it works
1. A simulator generates mostly normal readings with periodic patterns and noise.
2. Derived features describe relationships between channels.
3. Isolation Forest learns a baseline from synthetic data and predicts unusual samples.
4. The dashboard displays flagged readings for human review.

## Limitations
This is a **demonstration project**, not a certified industrial safety system. The model is trained on synthetic data; flagged readings indicate statistical unusualness, not a confirmed fault. Before real-world use, validate with representative site data, evaluate false-positive/false-negative rates, add sensor calibration and data-quality checks, and apply deployment-appropriate security.

## Future improvements
- MQTT or serial ingestion from a microcontroller.
- Rolling-window features and drift monitoring.
- SQLite/PostgreSQL storage.
- Authenticated users and alert delivery.
- Evaluation on labeled site data.

## Suggested GitHub topics
`artificial-intelligence` `anomaly-detection` `industrial-iot` `machine-learning` `python` `streamlit` `predictive-maintenance` `sensor-monitoring`

## Author
Add your name, LinkedIn, and demo link here.
