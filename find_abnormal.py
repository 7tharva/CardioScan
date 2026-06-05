import librosa, pickle, numpy as np, os, csv

model = pickle.load(open(r"C:\Users\Atharva\OneDrive\Documents\RHD_Detector\model\rhd_model.pkl", "rb"))

base = r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
folders = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]

results = []

for folder in folders:
    folder_path = os.path.join(base, folder)
    ref = os.path.join(folder_path, "REFERENCE.csv")
    if not os.path.exists(ref):
        continue
    with open(ref) as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[1].strip() != "1":
                continue
            wav = os.path.join(folder_path, row[0].strip() + ".wav")
            if not os.path.exists(wav):
                continue
            try:
                audio, sr = librosa.load(wav, sr=None)
                audio = audio - np.mean(audio)
                audio = librosa.effects.preemphasis(audio, coef=0.97)
                m = np.max(np.abs(audio))
                if m > 0: audio = audio / m
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                features = np.mean(mfccs, axis=1).reshape(1, -1)
                pred = model.predict(features)[0]
                prob = round(model.predict_proba(features)[0][1] * 100, 1)
                if pred == 1:
                    results.append((prob, wav))
                    print(f"FOUND: {wav} — {prob}%")
            except:
                pass

results.sort(reverse=True)
print("\n=== TOP 10 MOST CONFIDENT ABNORMAL FILES ===")
for prob, path in results[:10]:
    print(f"{prob}% — {path}")