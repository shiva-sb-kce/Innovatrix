import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🚀 Program Started...")

# 🔊 Feature extraction
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# 📂 Load dataset
def load_data():
    X = []
    y = []

    folders = ["dataset/human", "dataset/machine"]

    for label, folder in enumerate(folders):
        print(f"📁 Reading folder: {folder}")

        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            if file.endswith(".wav"):
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
            else:
                print(f"⚠️ Skipping non-wav file: {file}")

    print(f"✅ Total samples loaded: {len(X)}")
    return np.array(X), np.array(y)

# 📊 Load data
X, y = load_data()

if len(X) == 0:
    print("❌ No data found! Check your dataset.")
    exit()

# 🤖 Train model
print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 📈 Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 🔮 Prediction function
def predict(file):
    print(f"\n🔍 Predicting for: {file}")

    features = extract_features(file)
    if features is None:
        print("❌ Error extracting features")
        return

    probs = model.predict_proba([features])
    confidence = np.max(probs)
    prediction = model.predict([features])[0]

    if confidence >= 0.85:
        decision = "High Confidence ✅"
    elif confidence >= 0.65:
        decision = "Needs Review 🤔"
    else:
        decision = "Uncertain ⚠️"

    label = "Human Voice 🧍" if prediction == 0 else "Machine Voice 🤖"

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Decision: {decision}")

# 🎯 Test prediction (change file name if needed)
predict("dataset/machine/m1.wav")