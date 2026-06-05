import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import csv

print("Starting training...")

# Path to your dataset folders
training_folders = [
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training\training-a",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training\training-b",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training\training-c",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training\training-d",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training\training-e",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training\training-f",
]

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=500)
        # Remove DC offset
        audio = audio - np.mean(audio)
        # Preemphasis
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        # Normalise
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

X = []  # features
y = []  # labels

for folder in training_folders:
    reference_file = os.path.join(folder, "REFERENCE.csv")
    
    if not os.path.exists(reference_file):
        print(f"No REFERENCE.csv found in {folder}, skipping...")
        continue

    print(f"Processing folder: {folder}")

    with open(reference_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            
            filename = row[0].strip()
            label = row[1].strip()

            wav_path = os.path.join(folder, filename + ".wav")

            if not os.path.exists(wav_path):
                continue

            features = extract_features(wav_path)
            if features is None:
                continue

            if label == "1":
                y.append(1)  # abnormal
            elif label == "-1":
                y.append(0)  # normal
            else:
                continue

            X.append(features)
            print(f"Loaded: {filename} → {'ABNORMAL' if label == '1' else 'NORMAL'}")

print(f"\nTotal files loaded: {len(X)}")

# ---- Split into training and testing ----
X_array = np.array(X)
y_array = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} files, testing on {len(X_test)} files...")
print("Training model... this might take a minute!")

# ---- Train ----
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ---- Test accuracy ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 MODEL ACCURACY REPORT")
print(f"="*40)
print(f"Overall Accuracy: {round(accuracy * 100, 2)}%")
print(f"="*40)
print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

# ---- Save the model ----
model_path = r"C:\Users\Atharva\OneDrive\Documents\RHD_Detector\model\rhd_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model trained and saved!")