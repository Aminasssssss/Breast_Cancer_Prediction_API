import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("Breast Cancer Prediction")
st.markdown("Enter tumor measurements to predict **Malignant** or **Benign**.")
st.divider()

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** Random Forest Classifier
    **Dataset:** Breast Cancer Wisconsin (sklearn)
    **Task:** Binary classification
    - 0 → Malignant
    - 1 → Benign
    **Metrics:**
    - Accuracy: 94.74%
    - ROC-AUC: 99.40%
    - F1 Score: 95.83%
    """)
    st.divider()
    st.markdown("**ML System | Production Ready**")

st.subheader("Enter tumor features")

col1, col2, col3 = st.columns(3)

defaults = [17.99, 10.38, 122.8, 1001.0, 0.1184,
            0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399,
            0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622,
            0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

labels = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

features = []
for i, (label, default) in enumerate(zip(labels, defaults)):
    with [col1, col2, col3][i % 3]:
        val = st.number_input(label, value=float(default), format="%.5f", key=f"f{i}")
        features.append(val)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    try:
        resp = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)
        
        if resp.status_code == 422:
            st.error("Validation error: check input data format.")
            st.json(resp.json())
        elif resp.status_code == 503:
            st.error("Model is not loaded on the server.")
        elif resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            result = resp.json()
            diag = result["diagnosis"]
            prob = result["probability"]
            latency = result.get("latency_ms", 0)
            
            if diag == "benign":
                st.success(f"BENIGN — confidence {prob['benign']*100:.1f}%")
            else:
                st.error(f"MALIGNANT — confidence {prob['malignant']*100:.1f}%")
            
            st.caption(f"Prediction latency: {latency} ms")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Benign probability", f"{prob['benign']*100:.1f}%")
                st.progress(prob["benign"])
            with c2:
                st.metric("Malignant probability", f"{prob['malignant']*100:.1f}%")
                st.progress(prob["malignant"])
            
            if result.get("drift_detected"):
                st.warning("Data drift detected. Predictions may be less reliable.")
            
            with st.expander("Raw API response"):
                st.json(result)
                
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Run: make run-api")
    except Exception as e:
        st.error(f"Unexpected error: {e}")