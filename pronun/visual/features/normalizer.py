"""Enhanced 3D landmark normalization with depth normalization and statistical validation."""

import numpy as np

from pronun.config import ALL_LIP_INDICES, LEFT_LIP_CORNER, RIGHT_LIP_CORNER

# Indices within the extracted landmark array (not MediaPipe global indices)
_LEFT_IDX = ALL_LIP_INDICES.index(LEFT_LIP_CORNER)
_RIGHT_IDX = ALL_LIP_INDICES.index(RIGHT_LIP_CORNER)


def normalize_landmarks(landmarks: np.ndarray, enable_depth_norm: bool = True) -> np.ndarray:
    """Enhanced 3D mouth landmark normalization with depth processing.

    Processing steps:
    1. Centroid alignment: subtract mean position
    2. Mouth width scaling: normalize by lip corner distance  
    3. Depth normalization: standardize z-coordinates (optional)
    4. Outlier detection: validate landmark stability

    Args:
        landmarks: Array of shape (N, 3) with raw 3D landmarks.
        enable_depth_norm: Whether to apply depth normalization.

    Returns:
        Normalized landmarks of shape (N, 3).
    """
    # Step 1: Centroid alignment - remove translation variance
    centroid = landmarks.mean(axis=0)  # (3,)
    centered = landmarks - centroid

    # Step 2: Mouth width scaling - normalize by lip corner distance
    left = centered[_LEFT_IDX]
    right = centered[_RIGHT_IDX]
    width = np.linalg.norm(left - right)

    if width < 1e-8:
        # Degenerate case - return centered landmarks
        return centered

    scaled = centered / width

    # Step 3: Depth normalization (if enabled)
    if enable_depth_norm:
        z_coords = scaled[:, 2]
        z_std = np.std(z_coords)
        
        if z_std > 1e-8:
            # Normalize z-coordinates to unit variance
            z_mean = np.mean(z_coords)
            scaled[:, 2] = (z_coords - z_mean) / z_std
        
    return scaled


def get_mouth_width(landmarks: np.ndarray) -> float:
    """Compute 3D mouth width from raw landmarks."""
    left = landmarks[_LEFT_IDX]
    right = landmarks[_RIGHT_IDX]
    return float(np.linalg.norm(left - right))


def validate_landmarks(landmarks: np.ndarray, variance_threshold: float = 0.001) -> bool:
    """Validate landmark stability and detect outliers.
    
    Args:
        landmarks: Normalized landmarks array (N, 3).
        variance_threshold: Minimum variance threshold for stability.
        
    Returns:
        True if landmarks are valid and stable.
    """
    # Check for NaN or infinite values
    if not np.all(np.isfinite(landmarks)):
        return False
    
    # Check variance across landmarks (should not be too small)
    variance = np.var(landmarks, axis=0)
    if np.any(variance < variance_threshold):
        return False
        
    # Check for extreme outliers (z-score > 4)
    centered = landmarks - landmarks.mean(axis=0)
    std = landmarks.std(axis=0) + 1e-8
    z_scores = np.abs(centered / std)
    if np.any(z_scores > 4.0):
        return False
        
    return True


def normalize_sequence(landmark_seq: list[np.ndarray | None], 
                      enable_depth_norm: bool = True,
                      enable_validation: bool = True) -> list[np.ndarray | None]:
    """Enhanced sequence normalization with validation and statistical stability.
    
    Args:
        landmark_seq: List of landmark arrays or None.
        enable_depth_norm: Whether to apply depth normalization.
        enable_validation: Whether to validate landmark stability.
        
    Returns:
        List of normalized landmarks with invalid frames as None.
    """
    normalized_seq = []
    
    for lm in landmark_seq:
        if lm is None:
            normalized_seq.append(None)
            continue
            
        try:
            # Apply enhanced normalization
            normalized = normalize_landmarks(lm, enable_depth_norm=enable_depth_norm)
            
            # Validate stability if enabled
            if enable_validation and not validate_landmarks(normalized):
                normalized_seq.append(None)  # Mark as invalid
            else:
                normalized_seq.append(normalized)
                
        except Exception:
            # Handle any normalization errors
            normalized_seq.append(None)
    
    return normalized_seq


def get_sequence_statistics(landmark_seq: list[np.ndarray | None]) -> dict:
    """Compute statistics for a landmark sequence to monitor stability.
    
    Args:
        landmark_seq: List of landmark arrays.
        
    Returns:
        Dict with sequence statistics for monitoring distribution drift.
    """
    valid_landmarks = [lm for lm in landmark_seq if lm is not None]
    
    if not valid_landmarks:
        return {
            "valid_frames": 0,
            "total_frames": len(landmark_seq),
            "validity_ratio": 0.0,
            "mean_width": 0.0,
            "width_variance": 0.0,
        }
    
    # Compute mouth width statistics
    widths = [get_mouth_width(lm) for lm in valid_landmarks]
    
    return {
        "valid_frames": len(valid_landmarks),
        "total_frames": len(landmark_seq),
        "validity_ratio": len(valid_landmarks) / len(landmark_seq),
        "mean_width": float(np.mean(widths)),
        "width_variance": float(np.var(widths)),
        "width_stability": float(np.std(widths) / (np.mean(widths) + 1e-8)),
    }
