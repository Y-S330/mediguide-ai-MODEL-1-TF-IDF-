import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import re
import os

st.set_page_config(page_title="MediGuide AI", layout="wide")

BASE = os.path.dirname(__file__)

# ---------- CLEAN ----------
def _clean(t):
    t = str(t).lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------- LOADERS ----------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "symptoms_to_disease_model.pkl"))

@st.cache_data
def load_precautions():
    with open(os.path.join(BASE, "precautions_map.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_descriptions():
    df = pd.read_csv(os.path.join(BASE, "symptom_Description.csv"))
    return dict(zip(df["Disease"].apply(_clean), df["Description"]))

@st.cache_data
def load_symptoms():
    df = pd.read_csv(os.path.join(BASE, "DiseaseAndSymptoms.csv"))
    syms = set()
    for col in df.columns:
        if "Symptom" in col:
            syms.update(df[col].dropna().str.strip().tolist())
    return sorted(syms)

# ---------- LOAD ----------
model = load_model()
prec_map = load_precautions()
desc_map = load_descriptions()
symptom_list = load_symptoms()

# ---------- PREDICT ----------
def predict_topk(inp, k=5):
    inp = _clean(inp)
    if not inp:
        return []

    proba = model.predict_proba([inp])[0]
    top_idx = np.argsort(proba)[::-1][:k]

    return [(model.classes_[i], float(proba[i])) for i in top_idx]

def get_precautions(name):
    key = _clean(name)
    return prec_map.get(key, [])

def get_description(name):
    key = _clean(name)
    return desc_map.get(key, "No description available")

# ---------- UI ----------
st.title("MediGuide AI")

selected = st.multiselect("Select Symptoms", symptom_list)
text = st.text_area("Or type symptoms")

if st.button("Diagnose"):
    combined = " ".join(selected) + " " + text

    results = predict_topk(combined)

    if not results:
        st.warning("Enter symptoms")
    else:
        disease, conf = results[0]

        st.subheader(f"Predicted Disease: {disease}")
        st.write(f"Confidence: {conf*100:.2f}%")

        st.write("Description:")
        st.write(get_description(disease))

        st.write("Precautions:")
        for p in get_precautions(disease):
            st.write("-", p)