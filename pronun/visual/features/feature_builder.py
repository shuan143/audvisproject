"""Unified feature vector construction from normalized 3D mouth landmarks."""

import numpy as np
from scipy.spatial import ConvexHull

from pronun.config import (
    ALL_LIP_INDICES,
    LEFT_LIP_CORNER,
    LOWER_LIP_MID,
    OUTER_LIP_INDICES,
    RIGHT_LIP_CORNER,
    UPPER_LIP_MID,
)

# Indices within the extracted landmark array
_LEFT_IDX = ALL_LIP_INDICES.index(LEFT_LIP_CORNER)
_RIGHT_IDX = ALL_LIP_INDICES.index(RIGHT_LIP_CORNER)
_UPPER_IDX = ALL_LIP_INDICES.index(UPPER_LIP_MID)
_LOWER_IDX = ALL_LIP_INDICES.index(LOWER_LIP_MID)

# Upper lip indices (within outer lip, first half roughly)
_UPPER_LIP_OUTER = [ALL_LIP_INDICES.index(i) for i in OUTER_LIP_INDICES[:10]]
_LOWER_LIP_OUTER = [ALL_LIP_INDICES.index(i) for i in OUTER_LIP_INDICES[10:]]


def build_feature(normalized: np.ndarray) -> np.ndarray:
    """Build a unified feature vector from normalized 3D landmarks.

    Features:
        1. Mouth height H (3D distance upper↔lower lip midpoints)
        2. Opening ratio R = H / W (W=1 after normalization)
        3. Lip protrusion P (avg z-difference upper vs lower lip)
        4. 3D mouth area (convex hull area)
        5. Flattened normalized 3D landmarks [x1,y1,z1,...,xN,yN,zN]

    Args:
        normalized: Normalized landmarks of shape (N, 3).

    Returns:
        Feature vector as 1D array.
    """
    upper = normalized[_UPPER_IDX]
    lower = normalized[_LOWER_IDX]

    # 1. Mouth height
    height = float(np.linalg.norm(upper - lower))

    # 2. Opening ratio (width = 1 after normalization)
    ratio = height

    # 3. Lip protrusion (z-axis difference between upper and lower lip)
    upper_z = normalized[_UPPER_LIP_OUTER, 2].mean()
    lower_z = normalized[_LOWER_LIP_OUTER, 2].mean()
    protrusion = float(upper_z - lower_z)

    # 4. 3D mouth area via convex hull
    try:
        hull = ConvexHull(normalized[:, :2])  # 2D projection for area
        area = float(hull.volume)  # In 2D, volume = area
    except Exception:
        area = 0.0

    # 5. Flattened landmarks
    flat = normalized.flatten()

    # Concatenate all features
    geometric = np.array([height, ratio, protrusion, area], dtype=np.float64)
    return np.concatenate([geometric, flat])


def build_feature_sequence(
    normalized_seq: list[np.ndarray | None],
) -> list[np.ndarray]:
    """Build feature vectors with enhanced temporal features for a frame sequence.

    For each frame t, computes:
        f_t = build_feature(normalized_t) - static features
        Δf_t = f_t - f_{t-1} - first-order temporal derivatives  
        v_t = lip_movement_velocity(normalized_t, normalized_{t-1}) - velocity features
        output_t = concatenate(f_t, Δf_t, v_t)

    Frames where landmarks are None are skipped.

    Returns:
        List of enhanced feature vectors with velocity and temporal derivatives.
    """
    static_features = []
    normalized_valid = []
    valid_indices = []

    for i, norm in enumerate(normalized_seq):
        if norm is not None:
            static_features.append(build_feature(norm))
            normalized_valid.append(norm)
            valid_indices.append(i)

    if not static_features:
        return []

    result = []
    for i, feat in enumerate(static_features):
        # First-order temporal derivatives (deltas)
        if i == 0:
            delta = np.zeros_like(feat)
        else:
            delta = feat - static_features[i - 1]
        
        # Lip movement velocity features
        if i == 0:
            velocity = np.zeros(6)  # [vx, vy, vz for upper/lower lip midpoints]
        else:
            velocity = _compute_lip_velocity(normalized_valid[i], normalized_valid[i-1])
        
        # Enhanced feature vector: static + deltas + velocity
        enhanced_feat = np.concatenate([feat, delta, velocity])
        result.append(enhanced_feat)

    return result


def _compute_lip_velocity(current_landmarks: np.ndarray, 
                         previous_landmarks: np.ndarray) -> np.ndarray:
    """Compute lip movement velocity between consecutive frames.
    
    Args:
        current_landmarks: Current frame normalized landmarks (N, 3).
        previous_landmarks: Previous frame normalized landmarks (N, 3).
        
    Returns:
        6-dimensional velocity vector [upper_vx, upper_vy, upper_vz, 
                                      lower_vx, lower_vy, lower_vz].
    """
    # Velocity of key lip points (upper and lower midpoints)
    upper_curr = current_landmarks[_UPPER_IDX]
    upper_prev = previous_landmarks[_UPPER_IDX]
    upper_velocity = upper_curr - upper_prev
    
    lower_curr = current_landmarks[_LOWER_IDX]
    lower_prev = previous_landmarks[_LOWER_IDX]
    lower_velocity = lower_curr - lower_prev
    
    return np.concatenate([upper_velocity, lower_velocity])


def feature_dim(num_landmarks: int) -> int:
    """Compute expected feature dimension.

    Static: 4 geometric + 3*N landmark coords
    Deltas: Same dimensions as static features  
    Velocity: 6 dimensional lip velocity features
    Total: static + deltas + velocity
    """
    static = 4 + 3 * num_landmarks
    deltas = static
    velocity = 6
    return static + deltas + velocity
