import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
MODEL_PATH = "model_fixed.h5"
SCALER_PATH = "scaler.pkl"

SEQ_LEN = 10
THRESHOLD = 0.7
INPUT_FEATURES = 40

# =========================
# LOAD
# =========================
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# SINGLE
# =========================
def predict_single(features):

    try:
        if len(features) != INPUT_FEATURES:
            raise ValueError(f"Expected {INPUT_FEATURES}, got {len(features)}")

        x = np.array(features, dtype=float).reshape(1, -1)

        # scale
        x_scaled = scaler.transform(x)

        # sequence
        x_seq = np.repeat(x_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)

        prob = model.predict(x_seq, verbose=0)[0][0]

        label = "ATTACK" if prob >= THRESHOLD else "BENIGN"

        return {"label": label, "confidence": float(prob)}

    except Exception as e:
        return {"error": str(e)}


# =========================
# BATCH
# =========================
def predict_batch(X):

    try:
        X = np.array(X, dtype=float)

        if X.shape[1] != INPUT_FEATURES:
            raise ValueError(f"Expected {INPUT_FEATURES}, got {X.shape[1]}")

        X_scaled = scaler.transform(X)

        X_seq = np.repeat(X_scaled[:, np.newaxis, :], SEQ_LEN, axis=1)

        probs = model.predict(X_seq, verbose=0).flatten()

        results = []
        for p in probs:
            label = "ATTACK" if p >= THRESHOLD else "BENIGN"
            results.append({
                "label": label,
                "confidence": float(p)
            })

        return results

    except Exception as e:
        return {"error": str(e)}
