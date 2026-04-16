import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(page_title="MediGuide AI - TFIDF", layout="wide")

BASE = os.path.dirname(__file__)

# ================================
# FULL UI STYLE (FINAL)
# ================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #00FFB2;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 40px;
}

.card {
    background-color: #121212;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #00FFB2;
    box-shadow: 0px 0px 20px rgba(0,255,178,0.2);
    margin-top: 20px;
}

.stButton>button {
    width: 100%;
    background-color: #00FFB2;
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px;
}

.stTextArea textarea, .stMultiSelect div {
    background-color: #1e1e1e;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown('<div class="title">MediGuide AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">TF-IDF Disease Prediction</div>', unsafe_allow_html=True)

st.markdown("---")

# ================================
# CLEAN
# ================================
def clean(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t)

# ================================
# LOAD
# ================================
model = joblib.load(os.path.join(BASE, "symptoms_to_disease_model.pkl"))

with open(os.path.join(BASE, "precautions_map.pkl"), "rb") as f:
    prec_map = pickle.load(f)

desc_df = pd.read_csv(os.path.join(BASE, "symptom_Description.csv"))
desc_map = dict(zip(desc_df["Disease"].str.lower().str.replace(" ",""), desc_df["Description"]))

df = pd.read_csv(os.path.join(BASE, "DiseaseAndSymptoms.csv"))
symptoms = set()

for col in df.columns:
    if "Symptom" in col:
        symptoms.update(df[col].dropna())

symptom_list = sorted(symptoms)

# ================================
# INPUT UI
# ================================
col1, col2 = st.columns(2)

with col1:
    selected = st.multiselect("🧠 Select Symptoms", symptom_list)

with col2:
    text = st.text_area("✍️ Type Symptoms")

st.markdown("<br>", unsafe_allow_html=True)

center = st.columns([1,2,1])
with center[1]:
    run = st.button("🔍 Diagnose")

# ================================
# PREDICTION
# ================================
if run:
    combined = " ".join(selected) + " " + text

    if not combined.strip():
        st.warning("Please enter symptoms")
    else:
        with st.spinner("Analyzing symptoms..."):
            probs = model.predict_proba([combined])[0]
            i = np.argmax(probs)

            disease = model.classes_[i]
            conf = probs[i]

            key = disease.lower().replace(" ", "")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f"## 🧾 Prediction: {disease}")
        st.progress(int(conf * 100))
        st.write(f"Confidence: {conf*100:.2f}%")

        st.markdown("### 📄 Description")
        st.write(desc_map.get(key, "No description"))

        st.markdown("### 🛡️ Precautions")
        for p in prec_map.get(key, []):
            st.write("✔", p)

        st.markdown('</div>', unsafe_allow_html=True)