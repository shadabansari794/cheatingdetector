# 🚨 CheatingDetector — AI-Powered Real-Time Cheating Detection

> *Because if you're gonna cheat in an interview, at least expect a meme about it.* 😤

An AI-powered real-time cheating detection system built for online interviews. It monitors head movements, eye gaze, facial expressions, and even detects mobile phones — all through your webcam. And yes, when it catches you cheating, it slaps a **"Cheating karta hai tu"** meme on your face. 💀

---

## ✨ Features

| Feature | How It Works |
|---|---|
| 🧠 **Head Pose Tracking** | Detects if you're looking Left, Right, Up, or Down using MTCNN facial landmarks |
| 👁️ **Eye Gaze Detection** | Tracks pupil position to detect sneaky glances at notes |
| 📱 **Mobile Phone Detection** | Uses YOLOv4-Tiny to detect phones in the camera feed |
| 😬 **Expression Analysis** | Uses DeepFace to detect nervous expressions (fear, sadness) |
| 🎯 **Calibration** | Press `c` to set your neutral position for accurate detection |
| 😂 **Meme Overlay** | Displays a meme when cheating is detected — because why not |
| 📊 **Real-Time Dashboard** | On-screen status, debug values, and alert indicators |

---

## 🛠️ Tech Stack

- **Python 3.13+**
- **OpenCV** — Video capture, image processing, YOLOv4-Tiny DNN
- **MTCNN** — Face detection & facial landmark extraction
- **DeepFace** — Emotion/expression analysis
- **YOLOv4-Tiny** — Object detection (mobile phone)
- **TensorFlow** — Backend for DeepFace
- **NumPy** — Geometric calculations for head pose & gaze

---

## 📁 Project Structure

```
cheatingdetector/
├── main.py                  # Main application — orchestrates everything
├── gaze_detector.py         # Eye gaze direction detection
├── head_pose_estimator.py   # Head pose estimation (yaw/pitch)
├── expression_detector.py   # Facial expression analysis (DeepFace)
├── object_detector.py       # Mobile phone detection (YOLOv4-Tiny)
├── cheating_meme.jpg        # The legendary meme overlay 😂
├── yolov4-tiny.weights      # YOLOv4-Tiny model weights
├── yolov4-tiny.cfg          # YOLOv4-Tiny model config
├── coco.names               # COCO class labels
├── requirements.txt         # Python dependencies
└── test_mtcnn.py            # Quick test script for MTCNN
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/shadabansari794/cheatingdetector.git
cd cheatingdetector
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `c` | **Calibrate** — Look straight at the camera and press `c` to set your neutral position |
| `q` | **Quit** — Exit the application |

---

## 🔧 How It Works

### Detection Pipeline

```
Webcam Frame
    │
    ├──→ MTCNN Face Detection → Landmarks (eyes, nose, mouth)
    │       │
    │       ├──→ Head Pose Estimator (Nose vs Eye Midpoint projection)
    │       │       └──→ Yaw/Pitch delta from calibrated neutral
    │       │
    │       └──→ Gaze Detector (Pupil position in eye ROI)
    │               └──→ Smoothed gaze ratio (EMA)
    │
    ├──→ YOLOv4-Tiny → Mobile Phone Detection
    │
    └──→ DeepFace → Emotion Analysis (threaded, every 30 frames)

    All signals → Debounce Buffer → Alert / Meme Overlay 🎉
```

### Smoothing & Stability

- **Exponential Moving Average (EMA)** on head pose and gaze values to prevent jitter
- **Debounce buffer** — requires sustained suspicious behavior before triggering alerts
- **Calibration** — accounts for individual face geometry and camera angle

---

## ⚙️ Configuration

You can tune these values in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `alpha` | `0.4` | EMA smoothing factor (lower = smoother, higher = faster response) |
| `THRESHOLD_YAW` | `0.25` | Head turn sensitivity (lower = more sensitive) |
| `SUSPICIOUS_THRESHOLD_FRAMES` | `8` | Frames before alert triggers (~2 sec at 4 FPS) |
| Gaze thresholds | `0.35 / 0.65` | Gaze dead-zone (wider = less sensitive) |
| Buffer decay | `4` | How fast status returns to "Safe" |

---

## 📸 What It Looks Like

When you're being good:
> **Status: Safe** ✅

When you look away:
> **Status: Suspicious: Looking Away** 🔴

When you pull out your phone:
> **Status: Suspicious: Mobile Detected** 🔴 + 😂 Meme Overlay

---

## 🤝 Contributing

Feel free to open issues or PRs! Some ideas:

- [ ] Add audio detection (whispering detection)
- [ ] Add screen recording detection
- [ ] Web interface with Flask/FastAPI
- [ ] Multi-face tracking
- [ ] Configurable alert sounds
- [ ] Export suspicious timestamps to a report

---

## 📝 License

This project is open source. Use it, break it, meme it. 🫡

---

## 👨‍💻 Author

**Shadab Ansari**
- GitHub: [@shadabansari794](https://github.com/shadabansari794)

---

*Built with ☕, frustration, and a deep hatred for interview cheaters.* 😤
