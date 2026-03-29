import streamlit as st
import numpy as np
import pandas as pd
import io

from predict import predict_single, predict_batch

# =========================
# CONFIG
# =========================
NUM_FEATURES = 40

st.set_page_config(layout="wide")
st.title("🛡️ AI Intrusion Detection System")

# =========================
# INIT SESSION
# =========================
for i in range(NUM_FEATURES):
    if f"f_{i}" not in st.session_state:
        st.session_state[f"f_{i}"] = 0.0

# =========================
# AUTO FILL
# =========================
if st.button("⚡ Auto Fill Sample Data"):
    sample = np.random.uniform(0, 500, NUM_FEATURES)
    for i in range(NUM_FEATURES):
        st.session_state[f"f_{i}"] = float(sample[i])

# =========================
# INPUT UI
# =========================
st.subheader("Enter 40 Features")

cols = st.columns(5)

for i in range(NUM_FEATURES):
    with cols[i % 5]:
        st.number_input(f"F{i+1}", key=f"f_{i}")

# =========================
# SINGLE PREDICT
# =========================
if st.button("🚀 Predict Single Sample"):

    features = [st.session_state[f"f_{i}"] for i in range(NUM_FEATURES)]

    result = predict_single(features)

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("Result")

        if result["label"] == "ATTACK":
            st.error("🚨 ATTACK")
        else:
            st.success("✅ BENIGN")

        st.metric("Confidence", f"{result['confidence']:.4f}")

        # SAVE SINGLE RESULT
        single_df = pd.DataFrame([{
            **{f"f{i+1}": features[i] for i in range(NUM_FEATURES)},
            "prediction": result["label"],
            "confidence": result["confidence"]
        }])

        buffer = io.StringIO()
        single_df.to_csv(buffer, index=False)

        st.download_button(
            "⬇️ Download Single Result",
            buffer.getvalue(),
            "single_prediction.csv",
            "text/csv"
        )

# =========================
# CSV UPLOAD
# =========================
st.subheader("📂 Upload CSV (40 Features)")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.write("Detected columns:", df.shape[1])

    if df.shape[1] != NUM_FEATURES:
        st.error("CSV must have exactly 40 features")
        st.stop()

    results = predict_batch(df.values)

    preds = [r["label"] for r in results]
    confs = [r["confidence"] for r in results]

    df["prediction"] = preds
    df["confidence"] = confs

    # =========================
    # METRICS
    # =========================
    attack = preds.count("ATTACK")
    benign = preds.count("BENIGN")

    c1, c2 = st.columns(2)
    c1.metric("🚨 Attacks", attack)
    c2.metric("✅ Benign", benign)

    # =========================
    # BAR CHART
    # =========================
    chart = pd.DataFrame({
        "Class": ["ATTACK", "BENIGN"],
        "Count": [attack, benign]
    })

    st.subheader("📊 Prediction Distribution")
    st.bar_chart(chart.set_index("Class"))

    # =========================
    # PREVIEW
    # =========================
    st.dataframe(df.head())

    # =========================
    # DOWNLOAD
    # =========================
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    st.download_button(
        "⬇️ Download Full Results",
        buffer.getvalue(),
        "batch_predictions.csv",
        "text/csv"
    )