import streamlit as st
import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# 🎨 Page setup
st.set_page_config(page_title="Voice Classifier", layout="centered")

st.markdown("## 🎤 Human vs Machine Voice Detection")
st.markdown("---")
st.info("Upload audio file (.wav / .mp3 / .m4a)")

# 🔊 Feature extraction (NO conversion needed)
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, sr=None)
        audio = librosa.util.normalize(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# 📂 Load dataset
def load_data():
    X = []
    y = []

    folders = ["dataset/human", "dataset/machine"]

    for label, folder in enumerate(folders):
        if not os.path.exists(folder):
            st.warning(f"Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            if file.endswith((".wav", ".mp3", ".m4a")):
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)

    return np.array(X), np.array(y)

# 🧠 Train model
@st.cache_resource
def train_model():
    X, y = load_data()

    if len(X) == 0:
        return None

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# 🚀 Status
if model:
    st.success("Model Loaded Successfully 🚀")
else:
    st.error("Dataset missing or empty!")
    st.stop()

# 🎯 Upload file
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # ✅ Save file WITH extension (important fix)
    file_extension = uploaded_file.name.split(".")[-1]
    file_path = f"temp_input.{file_extension}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # 🔍 Extract features
    features = extract_features(file_path)

    if features is not None:
        probs = model.predict_proba([features])
        confidence = np.max(probs)
        prediction = model.predict([features])[0]

        # 🧠 Decision logic
        if confidence >= 0.85:
            decision = "High Confidence ✅"
        elif confidence >= 0.65:
            decision = "Needs Review 🤔"
        else:
            decision = "Uncertain ⚠️"

        label = "Human Voice 🧍" if prediction == 0 else "Machine Voice 🤖"

        # 📊 Output
        st.subheader("🔍 Result")
        st.write(f"Prediction: {label}")
        st.progress(float(confidence))
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"Decision: {decision}")