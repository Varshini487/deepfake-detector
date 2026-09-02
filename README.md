# 🔍 Deepfake Detector — AI-Generated Video Detection System

A hybrid CNN + LSTM deep learning system that detects deepfake and AI-manipulated videos by analyzing facial inconsistencies, frequency-domain artifacts, and temporal coherence across video frames.

## 🔑 Key Features

- **Hybrid CNN + LSTM architecture** — spatial analysis per frame + temporal consistency across frame sequences
- **Frequency-domain analysis** — FFT-based detection of GAN fingerprints and spectral artifacts
- **Spatial artifact detection** — LBP texture analysis, edge density, color coherence, blur scoring
- **Face detection & alignment** — automated face extraction from video frames
- **Confidence scoring** — weighted multi-feature scoring (0–1) with detailed artifact breakdown
- **Batch processing** — analyze multiple videos and generate CSV reports
- **Visualization** — heatmap overlays showing artifact concentration in face regions
- **Verdict system** — FAKE / REAL / UNCERTAIN based on per-frame analysis ratio

## 🧠 How It Works

1. **Frame Extraction**: Video is sampled at a configurable frame rate (default: every 5th frame)
2. **Face Detection**: Haar Cascade detects and crops faces from each frame
3. **Frequency Analysis**: 2D FFT computes spectral features (high-freq ratio, entropy, radial gradient)
4. **Spatial Analysis**: LBP histograms, edge density, and color statistics detect texture inconsistencies
5. **CNN Inference**: A convolutional neural network extracts deep spatial features from each face crop
6. **LSTM Temporal Analysis**: Frame-level features are fed into an LSTM to detect temporal flickering
7. **Confidence Fusion**: All features are combined with research-validated weights into a final score
8. **Verdict**: Per-frame scores are aggregated — >60% fake frames → FAKE, <30% → REAL, else UNCERTAIN

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Numerical Computation | NumPy |
| Face Detection | Haar Cascade (OpenCV) |
| Frequency Analysis | NumPy FFT |
| Spatial Analysis | LBP, Canny, Laplacian |
| Temporal Analysis | LSTM (PyTorch) |
| Reporting | CSV |

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Analyze a single video
```bash
python app.py --video path/to/video.mp4
```

### Batch analyze multiple videos
```bash
python app.py --batch video1.mp4 video2.mp4 video3.mp4
```

### Run demo with synthetic data
```bash
python app.py --demo
```

### In your Python code
```python
from app import DeepfakeDetector, Config

config = Config(frame_rate=5, confidence_threshold=0.5)
detector = DeepfakeDetector(config)

report = detector.analyze_video("suspicious_video.mp4")
print(f"Verdict: {report.verdict}")
print(f"Confidence: {report.avg_confidence}")
print(f"Fake frames: {report.fake_frames}/{report.frames_analyzed}")
```

## 📊 Output Example

```
============================================================
DEEPFAKE DETECTION REPORT
============================================================
Video:           test_video.mp4
Total Frames:    120
Frames Analyzed: 85
Fake Frames:     62
Avg Confidence:  0.7234
Max Confidence:  0.9512
Verdict:         FAKE
Processing Time: 12.45s

Frame Details:
 Frame     Time  Face  Fake   Conf
     0     0.00s   Yes   YES   0.7823
     1     0.20s   Yes   YES   0.8156
     2     0.40s   Yes    No   0.4123
     3     0.60s   Yes   YES   0.8901
     ...
============================================================
```

## 🏗️ Architecture

```
DeepfakeDetector
├── FaceDetector          → Haar Cascade face extraction
├── FrequencyAnalyzer     → FFT spectral feature extraction
│   ├── compute_fft_features()    → High-freq ratio, entropy, radial gradient
│   └── compute_spatial_features() → LBP, edges, color coherence
├── SpatialCNN            → 4-layer CNN for deep spatial features
├── TemporalLSTM          → 2-layer LSTM for temporal consistency
└── DeepfakeVisualizer    → Heatmap & annotation overlays
```

## 🔬 Feature Breakdown

| Feature | Weight | Fake Indicator |
|---------|--------|----------------|
| CNN Score | 35% | Higher = more likely fake |
| High-Frequency Ratio | 10% | Higher = GAN artifacts |
| Spectral Entropy | 10% | Higher = unnatural spectrum |
| Radial Gradient | 10% | Sharp drops = GAN fingerprint |
| LBP Entropy | 10% | Higher = texture inconsistency |
| Edge Density | 5% | Lower = blurred boundaries |
| Blur Score | 5% | Lower = synthetic smoothness |
| Hue/Value Std | 10% | Higher = color artifacts |
| Low-Freq Concentration | 5% | Lower = spectral anomalies |

## 📁 Project Structure

```
deepfake-detector/
├── app.py              # Main application with all modules
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── deepfake_report.csv # Auto-generated batch report
```

## 📝 License

MIT License — free for educational and research use.

## 💡 Interview Talking Points

1. **Multi-modal detection approach**: "Rather than relying on a single model, I built a hybrid system that combines CNN-based spatial analysis, FFT frequency-domain features, and LSTM temporal consistency — each catches different types of deepfake artifacts that the others would miss."

2. **Frequency-domain forensics**: "I implemented FFT-based spectral analysis to detect GAN fingerprints — deepfake generators leave characteristic traces in the Fourier spectrum that are invisible to the naked eye but detectable through radial gradient analysis and spectral entropy."

3. **Production-ready pipeline**: "The system includes batch processing with CSV reporting, configurable confidence thresholds, and a verdict system (FAKE/REAL/UNCERTAIN) — designed for real-world deployment where you need explainable results, not just a binary classification."
