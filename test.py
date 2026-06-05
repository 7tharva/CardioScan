import librosa, pickle, numpy as np

model = pickle.load(open(r"C:\Users\Atharva\OneDrive\Documents\RHD_Detector\model\rhd_model.pkl", "rb"))
folder = r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a"

for name in ["a0001","a0002","a0003","a0004","a0005","a0006","a0007","a0008","a0009","a0010"]:
    audio, sr = librosa.load(folder + "\\" + name + ".wav", sr=None)
    audio = audio - np.mean(audio)
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    m = np.max(np.abs(audio))
    if m > 0: audio = audio / m
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = np.mean(mfccs, axis=1).reshape(1, -1)
    pred = model.predict(features)[0]
    prob = round(model.predict_proba(features)[0][1] * 100, 1)
    label = "ABNORMAL" if pred == 1 else "NORMAL"
    print(f"{name}: {label} (abnormal prob: {prob}%)")