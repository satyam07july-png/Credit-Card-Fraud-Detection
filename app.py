import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging

# ================= LOAD =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= UI =================
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter transaction features (30 values)")

# ================= INPUT =================
inputs = []
cols = st.columns(3)

for i in range(30):
    with cols[i % 3]:
        val = st.number_input(f"V{i+1}", value=0.0, format="%.4f")
        inputs.append(val)

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= PREDICT =================
if st.button("Predict"):
    try:
        data = np.array(inputs).reshape(1, -1)
        data = scaler.transform(data)

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        st.subheader(f"Fraud Probability: {prob:.2%}")

        if pred == 1:
            st.error("🚨 Fraud Transaction Detected")
        else:
            st.success("✅ Normal Transaction")

        # ================= SHAP =================
        st.subheader("📊 Feature Impact")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, data, show=False)
        st.pyplot(fig)

        # ================= HISTORY =================
        st.session_state.history.append({
            "prediction": int(pred),
            "probability": float(prob)
        })

        logging.info("Prediction successful")

    except Exception as e:
        logging.error(f"Error: {e}")
        st.error("Something went wrong")

# ================= HISTORY DISPLAY =================
st.subheader("📜 Prediction History")

for i, h in enumerate(st.session_state.history[-5:]):
    st.write(f"{i+1}. Fraud: {h['prediction']} | Prob: {h['probability']:.2f}")

# ================= STYLE =================
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)