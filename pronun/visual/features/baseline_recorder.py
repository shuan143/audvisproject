"""Baseline mouth state recording and adaptive normalization for online usage."""

import numpy as np
from typing import Optional, List
import time

from pronun.visual.features.landmark_extractor import LandmarkExtractor
from pronun.visual.features.normalizer import normalize_sequence
from pronun.visual.features.feature_builder import build_feature_sequence
from pronun.workflow.camera import Camera


class BaselineRecorder:
    """Records baseline mouth state for adaptive normalization in online usage."""
    
    def __init__(self, camera: Camera, landmark_extractor: LandmarkExtractor):
        """Initialize baseline recorder.
        
        Args:
            camera: Camera instance for video capture.
            landmark_extractor: Landmark extractor for processing frames.
        """
        self.camera = camera
        self.landmark_extractor = landmark_extractor
        self.baseline_features: Optional[np.ndarray] = None
        self.baseline_landmarks: Optional[np.ndarray] = None
        
    def record_baseline(self, duration_seconds: float = 1.5) -> dict:
        """Record baseline mouth state for specified duration.
        
        User should maintain neutral mouth position during recording.
        
        Args:
            duration_seconds: Duration to record baseline (1-2 seconds recommended).
            
        Returns:
            Dict with baseline recording results.
        """
        print(f"Recording baseline mouth state for {duration_seconds:.1f} seconds...")
        print("Please maintain a neutral, relaxed mouth position.")
        
        # Record frames for baseline period
        baseline_frames = []
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            frame = self.camera.get_frame()
            if frame is not None:
                baseline_frames.append(frame)
                frame_count += 1
            time.sleep(0.033)  # ~30 FPS
        
        if not baseline_frames:
            return {
                "success": False,
                "error": "No frames captured during baseline recording",
                "frame_count": 0,
                "duration": 0.0
            }
        
        # Extract landmarks from baseline frames
        baseline_landmarks = self.landmark_extractor.extract_sequence(baseline_frames)
        valid_landmarks = [lm for lm in baseline_landmarks if lm is not None]
        
        if not valid_landmarks:
            return {
                "success": False,
                "error": "No valid landmarks detected in baseline frames",
                "frame_count": frame_count,
                "duration": time.time() - start_time
            }
        
        # Compute baseline statistics
        normalized_baseline = normalize_sequence(baseline_landmarks)
        baseline_features = build_feature_sequence(normalized_baseline)
        
        if not baseline_features:
            return {
                "success": False,
                "error": "No features extracted from baseline",
                "frame_count": frame_count,
                "duration": time.time() - start_time
            }
        
        # Store average baseline feature vector and landmark configuration
        self.baseline_features = np.mean(baseline_features, axis=0)
        valid_landmarks_array = np.array(valid_landmarks)
        self.baseline_landmarks = np.mean(valid_landmarks_array, axis=0)
        
        return {
            "success": True,
            "frame_count": frame_count,
            "valid_frames": len(valid_landmarks),
            "duration": time.time() - start_time,
            "baseline_feature_dim": self.baseline_features.shape[0],
            "baseline_ready": True
        }
    
    def apply_adaptive_normalization(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """Apply adaptive normalization: feature_adj = feature - baseline_feature.
        
        Args:
            features: List of feature vectors to normalize.
            
        Returns:
            List of adaptively normalized feature vectors.
            
        Raises:
            RuntimeError: If baseline has not been recorded.
        """
        if self.baseline_features is None:
            raise RuntimeError("Baseline must be recorded before applying adaptive normalization")
        
        normalized_features = []
        for feature in features:
            # Ensure feature dimensions match baseline
            if feature.shape[0] != self.baseline_features.shape[0]:
                # Handle dimension mismatch - pad or truncate
                min_dim = min(feature.shape[0], self.baseline_features.shape[0])
                feature_adj = feature[:min_dim] - self.baseline_features[:min_dim]
                
                # Pad with zeros if needed
                if feature.shape[0] > self.baseline_features.shape[0]:
                    padding = np.zeros(feature.shape[0] - min_dim)
                    feature_adj = np.concatenate([feature_adj, padding])
            else:
                feature_adj = feature - self.baseline_features
                
            normalized_features.append(feature_adj)
        
        return normalized_features
    
    def is_baseline_ready(self) -> bool:
        """Check if baseline has been successfully recorded."""
        return self.baseline_features is not None
    
    def get_baseline_info(self) -> dict:
        """Get information about the recorded baseline."""
        if not self.is_baseline_ready():
            return {"baseline_ready": False}
        
        return {
            "baseline_ready": True,
            "feature_dimension": self.baseline_features.shape[0],
            "landmark_shape": self.baseline_landmarks.shape if self.baseline_landmarks is not None else None,
            "baseline_feature_stats": {
                "mean": float(np.mean(self.baseline_features)),
                "std": float(np.std(self.baseline_features)),
                "min": float(np.min(self.baseline_features)),
                "max": float(np.max(self.baseline_features))
            }
        }
    
    def reset_baseline(self):
        """Reset/clear the recorded baseline."""
        self.baseline_features = None
        self.baseline_landmarks = None


class ExponentialMovingAverageFilter:
    """Exponential Moving Average temporal smoothing: S_t = α × X_t + (1-α) × S_{t-1}"""
    
    def __init__(self, alpha: float = 0.15):
        """Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0 < α < 1). Higher = more responsive.
                   α = 0.15 provides good balance of smoothing vs responsiveness.
        """
        self.alpha = alpha
        self.smoothed_state = None  # S_{t-1}
        
    def apply_filter(self, feature: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to feature vector.
        
        Args:
            feature: Current observation X_t (248D feature vector).
            
        Returns:
            Smoothed feature vector S_t.
        """
        if self.smoothed_state is None:
            # Initialize: S_0 = X_0
            self.smoothed_state = feature.copy()
            return self.smoothed_state
        
        # EMA update: S_t = α × X_t + (1-α) × S_{t-1}
        self.smoothed_state = (
            self.alpha * feature + 
            (1.0 - self.alpha) * self.smoothed_state
        )
        
        return self.smoothed_state.copy()
    
    def reset_filter(self):
        """Reset EMA state for new sequence."""
        self.smoothed_state = None


# Legacy compatibility class
class TemporalSmoothingFilter(ExponentialMovingAverageFilter):
    """Legacy compatibility wrapper for existing code."""
    
    def __init__(self, filter_type: str = "ema", window_size: int = 3, alpha: float = 0.15):
        """Initialize with EMA as default filter type."""
        super().__init__(alpha=alpha)
        self.filter_type = filter_type.lower()
        self.window_size = window_size