# CardioScan 🫀
### AI-Powered Rheumatic Heart Disease Screening Device
**🏆 1st Place — EWH × SUABE Designathon 2026, Fiji**

> A portable digital stethoscope that detects early signs of Rheumatic Heart Disease using machine learning — designed for community health workers in remote Fiji.

---

## What is CardioScan?

Rheumatic Heart Disease (RHD) is entirely preventable with a **$0.50 penicillin injection** — but only if caught early. In remote Fiji, there are no cardiologists, no echocardiography machines, and no affordable screening tools.

CardioScan puts AI-powered cardiac screening in the hands of community health workers — for under **$150**.

- ⏱ **15-second scan** — plug in, place on chest, get a result
- 🤖 **88.89% accuracy** — Random Forest model trained on 3,240 PhysioNet clinical recordings
- 🌐 **Browser-based** — runs in Chrome via Web Serial API, no app or installation needed
- 🔌 **Hardware + Software** — Arduino Nano captures heart sounds, Python + Flask processes them, ML model returns a CLEAR or REFER result

---

## Demo

![CardioScan Website](docs/demo.png)

**Live scan flow:**
1. Plug Arduino Nano into USB
2. Open website in Chrome
3. Click **Begin Scan** → select Arduino port → 15s recording begins
4. Audio samples sent to Flask server → ML model analyses → result displayed

**Secret demo shortcuts (for presentations):**
- `Shift + N` → runs a normal heart demo from PhysioNet dataset
- `Shift + A` → runs an abnormal/RHD demo (cycles through 5 verified files)

---

## Architecture

```
Browser (Chrome)
    │
    │  Web Serial API (USB)
    ▼
Arduino Nano + Microphone
    │  1,500 samples @ 100 samples/sec
    │  Serial → browser
    ▼
JavaScript (index.html)
    │  POST /api/analyse
    │  JSON: { samples: [...] }
    ▼
Flask Server (server.py)
    │  MFCC feature extraction (librosa)
    │  Random Forest inference (scikit-learn)
    ▼
Result: { prediction, normal_probability, abnormal_probability }
    │
    ▼
Website displays CLEAR / REFER
```

---

## ML Model

| Metric | Value |
|---|---|
| Dataset | PhysioNet CinC 2016 |
| Training recordings | 3,240 |
| Train / Test split | 2,592 / 648 |
| Overall accuracy | **88.89%** |
| Normal recall | **96%** |
| Normal precision | **91%** |
| Abnormal precision | **80%** |
| Abnormal recall | **64%** |

**Feature extraction:** 13 MFCC coefficients (mean across time axis) extracted using librosa, with DC offset removal, preemphasis filtering, and amplitude normalisation.

**Why Random Forest?** With 3,240 recordings, a deep neural network would overfit. Random Forest handles tabular feature vectors well at this dataset size, trains fast, and outputs calibrated probability scores — essential for a medical screening tool.

---

## Hardware

| Component | Specification |
|---|---|
| Microcontroller | Arduino Nano |
| Microphone | Electret microphone module (A0 pin) |
| Sample rate | 100 samples/sec (1,500 samples / 15s) |
| Baud rate | 9600 |
| Connection | USB Serial → Chrome Web Serial API |

**Known limitation:** The electret microphone captures at 100 Hz. The PhysioNet dataset was recorded at 44,100 Hz. Due to the Nyquist theorem, the current hardware only captures frequencies up to 50 Hz — below the 20–1,000 Hz cardiac range. A MEMS acoustic sensor at 4,000+ samples/sec would fully resolve this in a production version.

---

## Project Structure

```
cardioscan/
├── index.html          # Full-stack frontend (Web Serial API, canvas background, results UI)
├── server.py           # Flask backend (ML inference, demo endpoints)
├── train_model.py      # Model training pipeline
├── find_abnormal.py    # Scans dataset to find model-confirmed abnormal files
├── test.py             # Tests individual files against the model
├── model/
│   └── rhd_model.pkl   # Trained Random Forest model
└── README.md
```

---

## Getting Started

### Prerequisites
```
Python 3.8+
pip install flask flask-cors numpy librosa scikit-learn
```

### 1. Clone the repo
```bash
git clone https://github.com/7tharva/cardioscan.git
cd cardioscan
```

### 2. Download the dataset (for training only)
Download PhysioNet CinC 2016 from:
https://physionet.org/content/challenge-2016/1.0.0/

Place training folders at the path defined in `train_model.py`.

### 3. Train the model (or use the pre-trained one)
```bash
python train_model.py
```

### 4. Update demo file paths in server.py
Edit `DEMO_NORMAL_PATH` and `DEMO_ABNORMAL_PATHS` to point to your local PhysioNet files.

### 5. Start the server
```bash
python server.py
```

### 6. Open in Chrome
Go to `http://localhost:5000` — plug in your Arduino and click **Begin Scan**.

### 7. (Optional) Expose publicly with ngrok
```bash
ngrok http 5000
```
Update `SERVER_URL` in `index.html` with the ngrok URL.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML · CSS · Vanilla JavaScript · Canvas API |
| Hardware interface | Web Serial API (Chrome/Edge) |
| Backend | Python · Flask · flask-cors |
| ML | scikit-learn · Random Forest |
| Audio processing | librosa · NumPy |
| Tunnelling | ngrok |
| Hardware | Arduino Nano · Electret microphone |

---

## Limitations & Next Steps

**Current limitations:**
- Dataset is from international recordings — not Pacific-specific. Model accuracy on Fijian patients is unvalidated
- Electret microphone at 100 Hz misses most cardiac frequency content
- Requires laptop connection — not yet a fully standalone field device

**Roadmap:**
1. **Partner with CWMH Fiji** — collect local patient recordings from Colonial War Memorial Hospital, Suva
2. **MEMS microphone upgrade** — 4,000+ samples/sec to capture full cardiac range
3. **Standalone deployment** — migrate to Raspberry Pi or STM32 with TensorFlow Lite
4. **IP65 housing** — 3D-printed waterproof casing for tropical coastal environments
5. **Community pilot** — deploy across 3 remote island clinics in Fiji

---

## Research Context

This project was inspired by a published research paper on technology's role in improving healthcare access in low-income populations:

> Srivastava, A. (2024). *Technology and the Indian Healthcare System.* Innovapolis.
> https://innovapolis.ca/author/atharva-srivastava/

---

## References

- PhysioNet/CinC Challenge 2016 — https://physionet.org/content/challenge-2016/
- Wyber et al. (2024), *Rheumatic heart disease in the Pacific*, The Lancet Regional Health
- Watkins et al. (2017), *Global burden of rheumatic heart disease*, NEJM
- Menzies Pacific RHD Programme

---

## Acknowledgements

Built at the **EWH × SUABE Designathon 2026** — a designathon focused on engineering solutions for global health challenges in Fiji.

Thanks to the organisers **Sakura Brennan** and **Ayan Towhid** for running an exceptional event.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*CardioScan · Team HeartStoppers · University of Sydney · 2026*
