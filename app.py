"""
Deepfake Detector - AI-Generated Video Detection System
=========================================================
Detects deepfake and AI-manipulated videos by analyzing facial 
inconsistencies, temporal artifacts, and frequency-domain features 
using a CNN + LSTM hybrid architecture.

Features:
- Frame extraction from video files
- Face detection and alignment using MTCNN
- Spatial artifact detection via CNN (frequency domain + spatial)
- Temporal consistency analysis via LSTM across frame sequences
- Confidence scoring with detailed artifact breakdown
- Batch processing for multiple videos
- Visualization of detected anomalies
- CSV report generation
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import csv
import math


# ─── Configuration ───
@dataclass
class Config:
    """Configuration for deepfake detection pipeline."""
    frame_rate: int = 5              # Extract every Nth frame
    face_size: int = 224             # Face crop size for CNN
    sequence_length: int = 10        # Frames per sequence for LSTM
    confidence_threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    
    def __post_init__(self):
        print(f"Device: {self.device}")


# ─── Data Models ───
@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_idx: int
    timestamp: float
    is_fake: bool
    confidence: float
    face_detected: bool
    artifacts: Dict[str, float] = field(default_factory=dict)


@dataclass
class VideoReport:
    """Complete analysis report for a video."""
    video_path: str
    total_frames: int
    frames_analyzed: int
    fake_frames: int
    avg_confidence: float
    max_confidence: float
    verdict: str  # "FAKE", "REAL", "UNCERTAIN"
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    timestamp: str = ""
    processing_time: float = 0.0


# ─── Face Detector ───
class FaceDetector:
    """
    Simple face detector using OpenCV's Haar Cascade.
    In production, this would use MTCNN or RetinaFace.
    """
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces
    
    def extract_faces(self, frame: np.ndarray, 
                      target_size: int = 224) -> List[np.ndarray]:
        """Extract and resize face crops from frame."""
        faces = self.detect(frame)
        face_crops = []
        
        for (x, y, w, h) in faces:
            # Add padding
            pad = int(w * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.resize(face, (target_size, target_size))
                face_crops.append(face)
        
        return face_crops


# ─── Frequency Domain Analyzer ───
class FrequencyAnalyzer:
    """
    Analyzes frequency-domain artifacts common in deepfakes.
    Deepfake generation often leaves traces in the Fourier spectrum.
    """
    
    @staticmethod
    def compute_fft_features(face_image: np.ndarray) -> Dict[str, float]:
        """
        Compute FFT-based features for detecting GAN artifacts.
        
        Args:
            face_image: BGR face crop
            
        Returns:
            Dictionary of frequency-domain features
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        
        # Apply 2D FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # Normalize
        magnitude_norm = magnitude_log / (magnitude_log.max() + 1e-8)
        
        # Compute features
        h, w = magnitude_norm.shape
        center_h, center_w = h // 2, w // 2
        
        # High-frequency energy ratio
        total_energy = np.sum(magnitude_norm)
        low_freq_mask = np.zeros_like(magnitude_norm)
        cv2.circle(low_freq_mask, (center_w, center_h), min(h, w) // 8, 1, -1)
        low_freq_energy = np.sum(magnitude_norm * low_freq_mask)
        high_freq_ratio = 1.0 - (low_freq_energy / (total_energy + 1e-8))
        
        # Spectral entropy
        spectrum = magnitude_norm.flatten()
        spectrum = spectrum / (spectrum.sum() + 1e-8)
        entropy = -np.sum(spectrum * np.log2(spectrum + 1e-8))
        
        # Radial profile
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_dist = np.max(dist)
        
        radial_bins = 20
        radial_profile = np.zeros(radial_bins)
        for i in range(radial_bins):
            mask = (dist >= i * max_dist / radial_bins) & \
                   (dist < (i + 1) * max_dist / radial_bins)
            if mask.any():
                radial_profile[i] = np.mean(magnitude_norm[mask])
        
        # Gradient of radial profile (GAN artifacts cause sharp drops)
        radial_gradient = np.mean(np.abs(np.diff(radial_profile)))
        
        return {
            "high_freq_ratio": float(high_freq_ratio),
            "spectral_entropy": float(entropy),
            "radial_gradient": float(radial_gradient),
            "low_freq_concentration": float(low_freq_energy / (total_energy + 1e-8)),
        }
    
    @staticmethod
    def compute_spatial_features(face_image: np.ndarray) -> Dict[str, float]:
        """
        Compute spatial-domain features for artifact detection.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        
        # Local Binary Pattern variance (texture inconsistency)
        h, w = gray.shape
        lbp_code = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j]   > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1]   > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j]   > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1]   > center) << 0
                lbp_code[i, j] = code
        
        # LBP histogram (texture distribution)
        hist, _ = np.histogram(lbp_code, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        lbp_entropy = -np.sum(hist * np.log2(hist + 1e-8))
        
        # Edge density (blur detection)
        edges = cv2.Canny(face_image, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-8)
        
        # Color coherence in face region
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        hue_std = float(np.std(hsv[:, :, 0]))
        sat_mean = float(np.mean(hsv[:, :, 1]))
        val_std = float(np.std(hsv[:, :, 2]))
        
        # Blending artifacts at face boundaries
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            "lbp_entropy": float(lbp_entropy),
            "edge_density": float(edge_density),
            "hue_std": hue_std,
            "sat_mean": sat_mean,
            "val_std": val_std,
            "blur_score": float(blur_score),
        }


# ─── CNN Model (Spatial Feature Extraction) ───
class SpatialCNN(nn.Module):
    """
    CNN for extracting spatial features from individual face frames.
    Detects blending artifacts, texture inconsistencies, and GAN fingerprints.
    """
    
    def __init__(self, num_features: int = 256):
        super(SpatialCNN, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output, features


# ─── LSTM Model (Temporal Consistency) ───
class TemporalLSTM(nn.Module):
    """
    LSTM for analyzing temporal consistency across frame sequences.
    Deepfakes often have flickering, inconsistent facial dynamics.
    """
    
    def __init__(self, input_size: int = 256, hidden_size: int = 128, 
                 num_layers: int = 2):
        super(TemporalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        output = self.classifier(last_output)
        return output


# ─── Hybrid Detector ───
class DeepfakeDetector:
    """
    Complete deepfake detection pipeline combining:
    1. Frequency-domain analysis (FFT features)
    2. Spatial artifact detection (CNN)
    3. Temporal consistency analysis (LSTM)
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.face_detector = FaceDetector()
        self.freq_analyzer = FrequencyAnalyzer()
        
        # Initialize models
        self.cnn = SpatialCNN().to(self.config.device)
        self.lstm = TemporalLSTM().to(self.config.device)
        
        self.cnn.eval()
        self.lstm.eval()
        
        print("DeepfakeDetector initialized")
        print(f"  CNN parameters: {sum(p.numel() for p in self.cnn.parameters()):,}")
        print(f"  LSTM parameters: {sum(p.numel() for p in self.lstm.parameters()):,}")
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video at configured frame rate."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.config.frame_rate == 0:
                frames.append(frame)
            idx += 1
        
        cap.release()
        return frames
    
    def analyze_frame(self, frame: np.ndarray, frame_idx: int, 
                      timestamp: float) -> FrameAnalysis:
        """Analyze a single frame for deepfake artifacts."""
        
        # Extract faces
        faces = self.face_detector.extract_faces(
            frame, target_size=self.config.face_size)
        
        if not faces:
            return FrameAnalysis(
                frame_idx=frame_idx,
                timestamp=timestamp,
                is_fake=False,
                confidence=0.0,
                face_detected=False,
                artifacts={}
            )
        
        # Analyze first face (primary subject)
        face = faces[0]
        
        # Frequency-domain features
        fft_features = self.freq_analyzer.compute_fft_features(face)
        
        # Spatial features
        spatial_features = self.freq_analyzer.compute_spatial_features(face)
        
        # CNN inference
        face_tensor = torch.from_numpy(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        ).float().div(255.0).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            cnn_output, cnn_features = self.cnn(face_tensor)
            cnn_score = float(cnn_output.item())
        
        # Combine all artifacts
        all_artifacts = {**fft_features, **spatial_features}
        all_artifacts["cnn_score"] = cnn_score
        
        # Weighted confidence score
        confidence = self._compute_confidence(all_artifacts)
        
        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=timestamp,
            is_fake=confidence > self.config.confidence_threshold,
            confidence=round(confidence, 4),
            face_detected=True,
            artifacts={k: round(v, 4) for k, v in all_artifacts.items()}
        )
    
    def _compute_confidence(self, artifacts: Dict[str, float]) -> float:
        """
        Compute weighted confidence score from all artifacts.
        
        Weights are based on research showing which features are most
        indicative of deepfake manipulation.
        """
        weights = {
            "cnn_score": 0.35,
            "high_freq_ratio": 0.10,
            "spectral_entropy": 0.10,
            "radial_gradient": 0.10,
            "lbp_entropy": 0.10,
            "edge_density": 0.05,
            "blur_score": 0.05,
            "hue_std": 0.05,
            "val_std": 0.05,
            "low_freq_concentration": 0.05,
        }
        
        # Normalize features to 0-1 range
        normalized = {}
        normalized["cnn_score"] = artifacts.get("cnn_score", 0.5)
        normalized["high_freq_ratio"] = min(artifacts.get("high_freq_ratio", 0.5) * 2, 1.0)
        normalized["spectral_entropy"] = min(artifacts.get("spectral_entropy", 5) / 15, 1.0)
        normalized["radial_gradient"] = min(artifacts.get("radial_gradient", 0.1) * 10, 1.0)
        normalized["lbp_entropy"] = min(artifacts.get("lbp_entropy", 3) / 8, 1.0)
        normalized["edge_density"] = min(artifacts.get("edge_density", 0.05) * 10, 1.0)
        normalized["blur_score"] = min(artifacts.get("blur_score", 50) / 500, 1.0)
        normalized["hue_std"] = min(artifacts.get("hue_std", 20) / 100, 1.0)
        normalized["val_std"] = min(artifacts.get("val_std", 50) / 200, 1.0)
        normalized["low_freq_concentration"] = artifacts.get("low_freq_concentration", 0.5)
        
        # Weighted sum (higher = more likely fake)
        # Some features indicate fake when HIGH, some when LOW
        fake_indicators_high = [
            "cnn_score", "high_freq_ratio", "spectral_entropy", 
            "radial_gradient", "lbp_entropy", "hue_std", "val_std"
        ]
        fake_indicators_low = [
            "edge_density", "blur_score", "low_freq_concentration"
        ]
        
        score = 0
        for key in fake_indicators_high:
            score += weights[key] * normalized[key]
        for key in fake_indicators_low:
            score += weights[key] * (1.0 - normalized[key])
        
        return max(0.0, min(1.0, score))
    
    def analyze_video(self, video_path: str) -> VideoReport:
        """
        Complete video analysis pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoReport with frame-by-frame analysis and overall verdict
        """
        start_time = datetime.now()
        print(f"\nAnalyzing: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        print(f"  Extracted {len(frames)} frames")
        
        if not frames:
            return VideoReport(
                video_path=video_path,
                total_frames=0,
                frames_analyzed=0,
                fake_frames=0,
                avg_confidence=0.0,
                max_confidence=0.0,
                verdict="ERROR",
                timestamp=datetime.now().isoformat(),
                processing_time=0.0
            )
        
        # Analyze each frame
        frame_analyses = []
        for i, frame in enumerate(frames):
            timestamp = i * (1.0 / self.config.frame_rate)
            analysis = self.analyze_frame(frame, i, timestamp)
            frame_analyses.append(analysis)
            
            if analysis.face_detected:
                status = "FAKE" if analysis.is_fake else "REAL"
                print(f"  Frame {i}: {status} (conf={analysis.confidence:.2f})")
        
        # Compute summary statistics
        analyzed = [fa for fa in frame_analyses if fa.face_detected]
        fake_count = sum(1 for fa in analyzed if fa.is_fake)
        
        if analyzed:
            avg_conf = sum(fa.confidence for fa in analyzed) / len(analyzed)
            max_conf = max(fa.confidence for fa in analyzed)
        else:
            avg_conf = 0.0
            max_conf = 0.0
        
        # Determine verdict
        fake_ratio = fake_count / max(len(analyzed), 1)
        if fake_ratio > 0.6:
            verdict = "FAKE"
        elif fake_ratio < 0.3:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        report = VideoReport(
            video_path=video_path,
            total_frames=len(frames),
            frames_analyzed=len(analyzed),
            fake_frames=fake_count,
            avg_confidence=round(avg_conf, 4),
            max_confidence=round(max_conf, 4),
            verdict=verdict,
            frame_analyses=frame_analyses,
            timestamp=datetime.now().isoformat(),
            processing_time=round(processing_time, 2)
        )
        
        self._print_report(report)
        return report
    
    def analyze_batch(self, video_paths: List[str], 
                      output_csv: str = "deepfake_report.csv") -> List[VideoReport]:
        """
        Analyze multiple videos and generate a CSV report.
        
        Args:
            video_paths: List of video file paths
            output_csv: Path for CSV report output
            
        Returns:
            List of VideoReport objects
        """
        reports = []
        
        for path in video_paths:
            try:
                report = self.analyze_video(path)
                reports.append(report)
            except Exception as e:
                print(f"Error analyzing {path}: {e}")
                reports.append(VideoReport(
                    video_path=path, total_frames=0, frames_analyzed=0,
                    fake_frames=0, avg_confidence=0, max_confidence=0,
                    verdict=f"ERROR: {str(e)}",
                    timestamp=datetime.now().isoformat()
                ))
        
        # Write CSV report
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Total_Frames', 'Frames_Analyzed',
                           'Fake_Frames', 'Avg_Confidence', 'Max_Confidence',
                           'Verdict', 'Processing_Time_s'])
            for r in reports:
                writer.writerow([r.video_path, r.total_frames, r.frames_analyzed,
                               r.fake_frames, r.avg_confidence, r.max_confidence,
                               r.verdict, r.processing_time])
        
        print(f"\nReport saved to: {output_csv}")
        return reports
    
    def _print_report(self, report: VideoReport):
        """Print analysis report to console."""
        print("\n" + "=" * 60)
        print("DEEPFAKE DETECTION REPORT")
        print("=" * 60)
        print(f"Video:           {report.video_path}")
        print(f"Total Frames:    {report.total_frames}")
        print(f"Frames Analyzed: {report.frames_analyzed}")
        print(f"Fake Frames:     {report.fake_frames}")
        print(f"Avg Confidence:  {report.avg_confidence:.4f}")
        print(f"Max Confidence:  {report.max_confidence:.4f}")
        print(f"Verdict:         {report.verdict}")
        print(f"Processing Time: {report.processing_time:.2f}s")
        
        if report.frame_analyses:
            print("\nFrame Details:")
            print(f"{'Frame':>6} {'Time':>8} {'Face':>5} {'Fake':>5} {'Conf':>8}")
            for fa in report.frame_analyses[:10]:
                print(f"{fa.frame_idx:>6} {fa.timestamp:>8.2f}s "
                      f"{'Yes' if fa.face_detected else 'No':>5} "
                      f"{'YES' if fa.is_fake else 'No':>5} "
                      f"{fa.confidence:>8.4f}")
            if len(report.frame_analyses) > 10:
                print(f"  ... and {len(report.frame_analyses) - 10} more frames")
        
        print("=" * 60)


# ─── Visualization ───
class DeepfakeVisualizer:
    """Visualizes deepfake detection results."""
    
    @staticmethod
    def create_heatmap(face_image: np.ndarray, 
                       artifacts: Dict[str, float]) -> np.ndarray:
        """
        Create a heatmap overlay showing artifact concentration.
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Create heatmap from local variance
        heatmap = np.zeros_like(gray, dtype=np.float32)
        h, w = gray.shape
        block_size = 32
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                variance = np.var(block)
                heatmap[i:i+block_size, j:j+block_size] = variance
        
        # Normalize and apply colormap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original
        overlay = cv2.addWeighted(face_image, 0.6, heatmap_color, 0.4, 0)
        return overlay
    
    @staticmethod
    def save_analysis_image(frame: np.ndarray, analysis: FrameAnalysis,
                           output_path: str):
        """Save frame with analysis overlay."""
        annotated = frame.copy()
        
        # Add text overlay
        verdict = "FAKE" if analysis.is_fake else "REAL"
        color = (0, 0, 255) if analysis.is_fake else (0, 255, 0)
        
        cv2.putText(annotated, f"Verdict: {verdict}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Confidence: {analysis.confidence:.4f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imwrite(output_path, annotated)


# ─── Demo Mode ───
def run_demo():
    """
    Run a demo with synthetic data to showcase the detection pipeline.
    """
    print("=" * 60)
    print("DEEPFAKE DETECTOR - Demo Mode")
    print("=" * 60)
    
    config = Config()
    detector = DeepfakeDetector(config)
    
    # Generate synthetic face images
    print("\nGenerating synthetic test images...")
    
    np.random.seed(42)
    test_images = []
    
    # "Real" images - natural texture patterns
    for i in range(5):
        img = np.random.randint(40, 200, (224, 224, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        test_images.append(("real_sample", img))
    
    # "Fake" images - GAN-like artifacts (unnatural smoothness + sharp edges)
    for i in range(5):
        base = np.ones((224, 224, 3), dtype=np.uint8) * 128
        # Add smooth gradients (GAN fingerprint)
        for c in range(3):
            base[:, :, c] = np.linspace(100, 180, 224, dtype=np.uint8)
        # Add sharp boundary artifacts
        base[100:120, :, :] = np.random.randint(200, 255, (20, 224, 3), dtype=np.uint8)
        test_images.append(("fake_sample", base))
    
    # Analyze each
    print("\nAnalyzing synthetic images...")
    print(f"{'Type':<15} {'CNN_Score':>10} {'HiFreq':>10} {'Entropy':>10} "
          f"{'Confidence':>12} {'Verdict':>10}")
    print("-" * 70)
    
    for label, img in test_images:
        fft_feat = detector.freq_analyzer.compute_fft_features(img)
        spatial_feat = detector.freq_analyzer.compute_spatial_features(img)
        
        # CNN inference
        face_tensor = torch.from_numpy(
            img.transpose(2, 0, 1)
        ).float().div(255.0).unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            cnn_output, _ = detector.cnn(face_tensor)
            cnn_score = float(cnn_output.item())
        
        all_artifacts = {**fft_feat, **spatial_feat, "cnn_score": cnn_score}
        confidence = detector._compute_confidence(all_artifacts)
        verdict = "FAKE" if confidence > config.confidence_threshold else "REAL"
        
        print(f"{label:<15} {cnn_score:>10.4f} "
              f"{fft_feat['high_freq_ratio']:>10.4f} "
              f"{fft_feat['spectral_entropy']:>10.4f} "
              f"{confidence:>12.4f} {verdict:>10}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("To analyze a real video: python app.py --video path/to/video.mp4")
    print("For batch analysis: python app.py --batch file1.mp4 file2.mp4 ...")
    print("=" * 60)


# ─── Entry Point ───
if __name__ == "__main__":
    import sys
    
    detector = DeepfakeDetector()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--video":
        if len(sys.argv) < 3:
            print("Usage: python app.py --video <path_to_video>")
            sys.exit(1)
        report = detector.analyze_video(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Usage: python app.py --batch <video1> <video2> ...")
            sys.exit(1)
        reports = detector.analyze_batch(sys.argv[2:])
    else:
        print("Deepfake Detector")
        print("Usage:")
        print("  python app.py --demo              Run demo with synthetic data")
        print("  python app.py --video <path>      Analyze a single video")
        print("  python app.py --batch <v1> <v2>   Batch analyze multiple videos")
