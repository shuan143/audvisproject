"""Reference baseline from standard speakers for visual scoring."""

import numpy as np


class ReferenceBaseline:
    """Manages reference log-likelihood baselines from standard speakers.

    L_ref = average normalized log-likelihood from standard (native) speakers.
    Used to convert raw log-likelihoods into 0-100 scores.
    """

    def __init__(self):
        self._references: dict[str, float] = {}
        self._default_ref = -5.0  # conservative default

    def set_reference(self, word: str, log_likelihood_norm: float):
        """Set reference normalized log-likelihood for a word.

        Args:
            word: The reference word.
            log_likelihood_norm: Average L/T from standard speakers.
        """
        self._references[word] = log_likelihood_norm

    def get_reference(self, word: str) -> float:
        """Get reference log-likelihood for a word.

        Returns default if no specific reference exists.
        """
        return self._references.get(word, self._default_ref)

    def update_from_samples(self, word: str, log_likelihoods: list[float],
                            durations: list[int]):
        """Compute reference from multiple standard speaker samples.

        Args:
            word: The word being referenced.
            log_likelihoods: List of log P(O|HMM) from standard speakers.
            durations: List of sequence lengths T for each sample.
        """
        normalized = [ll / t for ll, t in zip(log_likelihoods, durations) if t > 0]
        if normalized:
            self._references[word] = float(np.mean(normalized))

    @property
    def default_reference(self) -> float:
        return self._default_ref

    @default_reference.setter
    def default_reference(self, value: float):
        self._default_ref = value

    def save(self, path: str):
        """Save references to disk."""
        np.savez(path, references=self._references, default=self._default_ref)

    def load(self, path: str):
        """Load references from disk."""
        data = np.load(path, allow_pickle=True)
        self._references = dict(data["references"].item())
        self._default_ref = float(data["default"])
