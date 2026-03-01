"""Universal reference baseline for speaker-independent visual scoring."""

import numpy as np
from pathlib import Path


class UniversalReferenceBaseline:
    """Universal reference baseline for speaker-independent visual scoring.

    Stores universal statistical parameters (μ_ref, σ_ref) calibrated from 
    GRID corpus validation data for the research-based scoring formula:
    Score = clamp(80 + 10 × (L_norm - μ_ref) / σ_ref, 0, 100)
    
    Key principle: Same reference statistics used for all speakers.
    """

    def __init__(self):
        # Universal parameters for all speakers
        self._mu_ref = -5.0  # Will be calibrated from GRID validation data
        self._sigma_ref = 1.0  # Will be calibrated from GRID validation data
        self._calibrated = False

    def calibrate_from_validation_data(self, validation_samples):
        """Calibrate universal reference from GRID validation speakers.
        
        SEPARATE calibration phase - uses trained HMMs on validation data
        to compute universal reference statistics for speaker-independent scoring.

        Args:
            validation_samples: List of (hmm, feature_sequences) from validation speakers.
        """
        log_likelihoods_norm = []
        
        for hmm, feature_sequences in validation_samples:
            for features in feature_sequences:
                if len(features) == 0:
                    continue
                    
                # Use trained HMM to compute likelihood
                L = hmm.forward(features)
                
                # Normalize by sequence length: L_norm = L / T
                L_norm = L / len(features)
                log_likelihoods_norm.append(L_norm)
        
        if len(log_likelihoods_norm) >= 2:
            # Universal statistics from validation speakers
            self._mu_ref = float(np.mean(log_likelihoods_norm))
            self._sigma_ref = float(np.std(log_likelihoods_norm))
            
            # Prevent division by zero
            self._sigma_ref = max(self._sigma_ref, 0.1)
            self._calibrated = True
        else:
            raise RuntimeError("Insufficient validation data for calibration")

    def get_universal_statistics(self) -> dict[str, float]:
        """Return universal reference statistics for all speakers.
        
        Returns:
            Dict with 'mu' and 'sigma' keys for universal baseline.
        """
        return {"mu": self._mu_ref, "sigma": self._sigma_ref}

    @property
    def is_calibrated(self) -> bool:
        """Check if universal baseline has been calibrated."""
        return self._calibrated
    
    @property
    def mu_ref(self) -> float:
        """Universal mean reference."""
        return self._mu_ref
    
    @property
    def sigma_ref(self) -> float:
        """Universal standard deviation reference."""
        return self._sigma_ref

    def save(self, path: str | Path):
        """Save universal reference parameters to disk."""
        np.savez(
            path, 
            mu_ref=self._mu_ref, 
            sigma_ref=self._sigma_ref,
            calibrated=self._calibrated
        )

    def load(self, path: str | Path):
        """Load universal reference parameters from disk."""
        data = np.load(path, allow_pickle=True)
        self._mu_ref = float(data["mu_ref"])
        self._sigma_ref = float(data["sigma_ref"])
        self._calibrated = bool(data["calibrated"])


# Legacy compatibility class
class ReferenceBaseline(UniversalReferenceBaseline):
    """Legacy compatibility wrapper for existing code."""
    
    def __init__(self):
        super().__init__()
        # Compatibility with old interface
        self._default_stats = {"mu": -5.0, "sigma": 1.0}
        self.default_statistics = self._default_stats
    
    def get_statistics(self, word: str = "") -> dict[str, float]:
        """Legacy interface - returns universal statistics regardless of word."""
        return self.get_universal_statistics()
    
    def set_statistics(self, word: str, mu: float, sigma: float):
        """Legacy interface - sets universal statistics."""
        self._mu_ref = mu
        self._sigma_ref = sigma
        self._calibrated = True
    
    def update_from_samples(self, word: str, log_likelihoods: list[float], sequence_lengths: list[int]):
        """Legacy interface - updates statistics from samples.
        
        Args:
            word: Word identifier (ignored in universal baseline).
            log_likelihoods: List of raw log-likelihoods.
            sequence_lengths: List of sequence lengths for normalization.
        """
        # Normalize by sequence length: L_norm = L / T
        normalized_likelihoods = [ll / length for ll, length in zip(log_likelihoods, sequence_lengths)]
        
        if len(normalized_likelihoods) >= 2:
            self._mu_ref = float(np.mean(normalized_likelihoods))
            self._sigma_ref = float(np.std(normalized_likelihoods))
            self._sigma_ref = max(self._sigma_ref, 0.1)  # minimum sigma constraint
            self._calibrated = True
        elif len(normalized_likelihoods) == 1:
            self._mu_ref = float(normalized_likelihoods[0])
            self._sigma_ref = 1.0  # default when only one sample
            self._calibrated = True
