from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import numpy as np
import librosa
import pickle
import os

app = Flask(__name__, static_folder=os.getcwd())
CORS(app)

MODEL_PATH = r"C:\Users\Atharva\OneDrive\Documents\RHD_Detector\model\rhd_model.pkl"
DEMO_NORMAL_PATH   = r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a\a0008.wav"
import random

DEMO_ABNORMAL_PATHS = [
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a\a0322.wav",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a\a0045.wav",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-f\f0060.wav",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-f\f0068.wav",
    r"C:\Users\Atharva\Desktop\All files Designathon\file\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-e\e02085.wav",
]
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded!")

def process_audio(audio, sr):
    audio = audio - np.mean(audio)
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = np.mean(mfccs, axis=1).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return {
        'prediction': 'NORMAL' if prediction == 0 else 'RHD',
        'normal_probability': round(float(probabilities[0]) * 100, 1),
        'abnormal_probability': round(float(probabilities[1]) * 100, 1),
    }

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/api/analyse', methods=['POST'])
def analyse():
    try:
        data = request.get_json()
        samples = data['samples']
        n = len(samples)
        audio = np.array(samples, dtype=float)
        audio = audio / 1023.0
        sr = 500 if n >= 5000 else 100
        result = process_audio(audio, sr)
        result['samples_collected'] = n
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/<mode>', methods=['POST'])
def demo(mode):
    try:
        if mode == 'normal':
            path = DEMO_NORMAL_PATH
        elif mode == 'abnormal':
            path = random.choice(DEMO_ABNORMAL_PATHS)
        else:
            return jsonify({'error': 'Invalid mode'}), 400
        audio, sr = librosa.load(path, sr=None)
        result = process_audio(audio, sr)
        result['samples_collected'] = len(audio)
        result['demo_mode'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*40)
    print("CardioScan running at http://localhost:5000")
    print("="*40)
    app.run(debug=False, port=5000, use_reloader=False)