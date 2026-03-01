"""Visual pronunciation scorer using research-based statistical scoring."""

import numpy as np

from pronun.visual.scoring.hmm import GaussianHMM
from pronun.visual.scoring.reference import ReferenceBaseline, UniversalReferenceBaseline


class VisualScorer:
    """Converts HMM log-likelihood into a 0-100 pronunciation score using research formula.

    Research-based scoring pipeline:
        1. L = log P(O | HMM) via Forward Algorithm  
        2. L_norm = L / T (time normalization)
        3. Score = clamp(80 + 10 × (L_norm - μ_ref) / σ_ref, 0, 100)
    
    Uses universal reference baseline (μ_ref, σ_ref) for speaker-independent scoring.
    """

    def __init__(self, reference: ReferenceBaseline | UniversalReferenceBaseline | None = None):
        self.reference = reference or ReferenceBaseline()

    def score(
        self,
        hmm: GaussianHMM,
        observations: np.ndarray,
        viseme_sequence: list[int] = None,
    ) -> dict:
        """Score an observation sequence against an HMM using research formula.

        Args:
            hmm: Trained HMM for the target viseme sequence.
            observations: Feature vectors, shape (T, feature_dim).
            viseme_sequence: Viseme sequence for output (optional).

        Returns:
            Dict with 'log_likelihood', 'log_likelihood_norm', 'viseme_sequence',
            'mu_ref', 'sigma_ref', 'score_raw', 'score', 'confidence'.
        """
        T = len(observations)
        if T == 0:
            stats = self.reference.get_universal_statistics()
            return {
                "log_likelihood": -np.inf,
                "log_likelihood_norm": -np.inf,
                "viseme_sequence": viseme_sequence or [],
                "mu_ref": stats["mu"],
                "sigma_ref": stats["sigma"],
                "score_raw": 0.0,
                "score": 0.0,
                "confidence": 0.0,
            }

        # Forward Algorithm: L = log P(O | λ)
        log_likelihood = hmm.forward(observations)
        
        # Time normalization: L_norm = L / T
        log_likelihood_norm = log_likelihood / T

        # Universal reference statistics (speaker-independent)
        stats = self.reference.get_universal_statistics()
        mu_ref = stats["mu"]
        sigma_ref = stats["sigma"]

        # Research scoring formula: Score = clamp(80 + 10 × (L_norm - μ_ref) / σ_ref, 0, 100)
        score_raw = 80.0 + 10.0 * (log_likelihood_norm - mu_ref) / sigma_ref
        score = max(0.0, min(100.0, score_raw))  # Clamp to [0, 100]
        
        # Confidence based on statistical distance and sequence length
        z_score = abs((log_likelihood_norm - mu_ref) / sigma_ref)
        confidence = np.exp(-0.5 * z_score) * min(1.0, T / 20.0)

        return {
            "log_likelihood": log_likelihood,
            "log_likelihood_norm": log_likelihood_norm,
            "viseme_sequence": viseme_sequence or [],
            "mu_ref": mu_ref,
            "sigma_ref": sigma_ref,
            "score_raw": score_raw,
            "score": score,
            "confidence": confidence,
        }

    def build_hmm(
        self,
        viseme_sequence: list[int],
        viseme_observations: dict[int, np.ndarray],
        feature_dim: int,
    ) -> GaussianHMM:
        """Build an HMM for a specific viseme sequence.

        Args:
            viseme_sequence: Target sequence of viseme IDs.
            viseme_observations: Dict mapping viseme_id → training observations
                                 of shape (num_obs, feature_dim).
            feature_dim: Dimensionality of feature vectors.

        Returns:
            Trained GaussianHMM ready for scoring.
        """
        num_states = len(viseme_sequence)
        hmm = GaussianHMM(num_states, feature_dim)

        for state_idx, viseme_id in enumerate(viseme_sequence):
            if viseme_id in viseme_observations:
                obs = viseme_observations[viseme_id]
                hmm.train_emissions(state_idx, obs)

        return hmm
