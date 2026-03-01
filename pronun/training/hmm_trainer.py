"""Maximum Likelihood Estimation training for HMM parameters using LRS2 dataset."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json
from collections import defaultdict
import time

from pronun.data.lrs2_dataset import LRS2Dataset, LRS2VideoProcessor
from pronun.visual.features.landmark_extractor import LandmarkExtractor
from pronun.visual.features.normalizer import normalize_sequence
from pronun.visual.features.feature_builder import build_feature_sequence
from pronun.visual.scoring.hmm import GaussianHMM
from pronun.visual.scoring.reference import ReferenceBaseline


class HMMTrainer:
    """Maximum Likelihood Estimation trainer for HMM parameters from LRS2 dataset."""
    
    def __init__(self, dataset: LRS2Dataset, landmark_extractor: LandmarkExtractor):
        """Initialize HMM trainer.
        
        Args:
            dataset: LRS2Dataset instance for training data.
            landmark_extractor: Feature extraction pipeline.
        """
        self.dataset = dataset
        self.landmark_extractor = landmark_extractor
        self.video_processor = LRS2VideoProcessor()
        
        # Training state
        self.viseme_observations: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.training_statistics: Dict[str, Dict[str, float]] = {}
        self.feature_dimension = None
        
        # Statistical stability tracking
        self.training_log_likelihoods: List[float] = []
        self.training_durations: List[int] = []
        
    def extract_features_from_dataset(self, max_samples: Optional[int] = None,
                                    skip_processing_errors: bool = True) -> Dict[str, any]:
        """Extract visual features from LRS2 dataset for HMM training.
        
        Args:
            max_samples: Maximum number of samples to process (None for all).
            skip_processing_errors: Skip samples with processing errors.
            
        Returns:
            Dict with extraction results and statistics.
        """
        print("Starting feature extraction from LRS2 dataset...")
        start_time = time.time()
        
        self.viseme_observations.clear()
        processed_samples = 0
        successful_samples = 0
        failed_samples = 0
        
        # Process dataset samples
        for i, (video_path, transcript, viseme_seq, speaker_id) in enumerate(self.dataset.get_samples()):
            if max_samples and processed_samples >= max_samples:
                break
                
            processed_samples += 1
            
            if processed_samples % 100 == 0:
                print(f"Processing sample {processed_samples}: {video_path.name}")
            
            try:
                # Extract features from video
                features = self._extract_video_features(video_path, viseme_seq)
                
                if features and len(features) > 0:
                    # Group features by viseme for MLE training
                    self._group_features_by_viseme(features, viseme_seq)
                    successful_samples += 1
                else:
                    if not skip_processing_errors:
                        print(f"Warning: No features extracted from {video_path}")
                    failed_samples += 1
                    
            except Exception as e:
                failed_samples += 1
                if not skip_processing_errors:
                    print(f"Error processing {video_path}: {e}")
                    raise
                else:
                    continue
        
        # Set feature dimension from first successful extraction
        if self.viseme_observations:
            first_viseme = next(iter(self.viseme_observations.keys()))
            if self.viseme_observations[first_viseme]:
                self.feature_dimension = self.viseme_observations[first_viseme][0].shape[0]
        
        extraction_time = time.time() - start_time
        
        results = {
            "processed_samples": processed_samples,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "extraction_time_seconds": extraction_time,
            "feature_dimension": self.feature_dimension,
            "visemes_with_data": len(self.viseme_observations),
            "total_feature_vectors": sum(len(obs) for obs in self.viseme_observations.values()),
            "viseme_data_counts": {vid: len(obs) for vid, obs in self.viseme_observations.items()}
        }
        
        print(f"Feature extraction completed in {extraction_time:.1f}s")
        print(f"Successfully processed {successful_samples}/{processed_samples} samples")
        print(f"Feature dimension: {self.feature_dimension}")
        print(f"Visemes with training data: {len(self.viseme_observations)}")
        
        return results
    
    def _extract_video_features(self, video_path: Path, viseme_seq: List[int]) -> List[np.ndarray]:
        """Extract visual features from a single video.
        
        Args:
            video_path: Path to video file.
            viseme_seq: Expected viseme sequence for alignment.
            
        Returns:
            List of feature vectors.
        """
        # Load video frames
        frames = self.video_processor.load_video_frames(video_path, max_frames=150)  # Limit for memory
        
        if not frames:
            return []
        
        # Extract landmarks
        landmarks = self.landmark_extractor.extract_sequence(frames)
        
        # Normalize and build features
        normalized = normalize_sequence(landmarks)
        features = build_feature_sequence(normalized)
        
        return features
    
    def _group_features_by_viseme(self, features: List[np.ndarray], viseme_seq: List[int]):
        """Group feature vectors by their corresponding viseme labels.
        
        Args:
            features: List of feature vectors from video.
            viseme_seq: Corresponding viseme sequence.
        """
        if not features or not viseme_seq:
            return
        
        # Align features with viseme sequence using simple temporal alignment
        # For robust training, we use uniform temporal segmentation
        feature_length = len(features)
        viseme_length = len(viseme_seq)
        
        if feature_length == 0 or viseme_length == 0:
            return
        
        # Map features to visemes using uniform segmentation
        features_per_viseme = max(1, feature_length // viseme_length)
        
        for i, viseme_id in enumerate(viseme_seq):
            # Determine feature indices for this viseme
            start_idx = i * features_per_viseme
            end_idx = min((i + 1) * features_per_viseme, feature_length)
            
            # Add features for this viseme
            for feat_idx in range(start_idx, end_idx):
                if feat_idx < len(features):
                    self.viseme_observations[viseme_id].append(features[feat_idx])
    
    def train_hmm_parameters(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Train HMM emission parameters using Maximum Likelihood Estimation.
        
        Returns:
            Dict mapping viseme_id to trained parameters (mean, covariance).
        """
        if not self.viseme_observations:
            raise RuntimeError("No training data available. Run extract_features_from_dataset() first.")
        
        print("Training HMM parameters using Maximum Likelihood Estimation...")
        
        trained_parameters = {}
        
        for viseme_id, observations in self.viseme_observations.items():
            if len(observations) < 2:
                print(f"Warning: Insufficient data for viseme {viseme_id} ({len(observations)} samples)")
                continue
            
            # Convert list to numpy array
            obs_array = np.array(observations)
            
            # Maximum Likelihood Estimation for Gaussian parameters
            mean = np.mean(obs_array, axis=0)
            cov = np.cov(obs_array, rowvar=False)
            
            # Regularization for numerical stability
            if cov.ndim == 0:  # Single feature case
                cov = np.array([[max(cov, 1e-6)]])
            else:
                # Add regularization to diagonal
                cov += 1e-6 * np.eye(cov.shape[0])
            
            trained_parameters[viseme_id] = {
                "mean": mean,
                "covariance": cov,
                "num_samples": len(observations)
            }
            
            print(f"Trained viseme {viseme_id}: {len(observations)} samples, "
                  f"mean shape: {mean.shape}, cov shape: {cov.shape}")
        
        print(f"Successfully trained parameters for {len(trained_parameters)} visemes")
        
        return trained_parameters
    
    def compute_training_statistics(self, trained_parameters: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """Compute training dataset statistics for pronunciation scoring.
        
        Args:
            trained_parameters: Trained HMM parameters.
            
        Returns:
            Dict with training statistics (mu, sigma).
        """
        print("Computing training dataset statistics...")
        
        log_likelihoods = []
        durations = []
        
        # Compute log-likelihood for each training sample
        sample_count = 0
        for i, (video_path, transcript, viseme_seq, speaker_id) in enumerate(self.dataset.get_samples()):
            if sample_count >= 1000:  # Limit for efficiency
                break
                
            try:
                # Extract features
                features = self._extract_video_features(video_path, viseme_seq)
                if not features or not viseme_seq:
                    continue
                
                # Build HMM with trained parameters
                hmm = self._build_hmm_from_parameters(viseme_seq, trained_parameters)
                
                # Compute log-likelihood
                obs_array = np.array(features)
                log_likelihood = hmm.forward(obs_array)
                
                log_likelihoods.append(log_likelihood)
                durations.append(len(features))
                sample_count += 1
                
                if sample_count % 100 == 0:
                    print(f"Processed {sample_count} samples for statistics")
                    
            except Exception as e:
                print(f"Warning: Error computing statistics for {video_path}: {e}")
                continue
        
        if not log_likelihoods:
            raise RuntimeError("No valid samples for computing training statistics")
        
        # Compute normalized log-likelihoods
        normalized_ll = [ll / dur for ll, dur in zip(log_likelihoods, durations) if dur > 0]
        
        if not normalized_ll:
            raise RuntimeError("No valid normalized log-likelihoods computed")
        
        # Training statistics
        mu = float(np.mean(normalized_ll))
        sigma = float(np.std(normalized_ll))
        sigma = max(sigma, 0.1)  # Minimum sigma for numerical stability
        
        self.training_statistics = {
            "mu": mu,
            "sigma": sigma,
            "num_samples": len(normalized_ll),
            "min_log_likelihood": float(np.min(normalized_ll)),
            "max_log_likelihood": float(np.max(normalized_ll))
        }
        
        print(f"Training statistics computed from {len(normalized_ll)} samples:")
        print(f"  μ (mean): {mu:.4f}")
        print(f"  σ (std):  {sigma:.4f}")
        print(f"  Range: [{np.min(normalized_ll):.4f}, {np.max(normalized_ll):.4f}]")
        
        return self.training_statistics
    
    def _build_hmm_from_parameters(self, viseme_seq: List[int], 
                                 trained_parameters: Dict[int, Dict[str, np.ndarray]]) -> GaussianHMM:
        """Build HMM from trained parameters.
        
        Args:
            viseme_seq: Viseme sequence.
            trained_parameters: Trained emission parameters.
            
        Returns:
            Configured GaussianHMM.
        """
        if not self.feature_dimension:
            raise RuntimeError("Feature dimension not determined")
        
        hmm = GaussianHMM(len(viseme_seq), self.feature_dimension)
        
        for state_idx, viseme_id in enumerate(viseme_seq):
            if viseme_id in trained_parameters:
                params = trained_parameters[viseme_id]
                hmm.set_emission_params(state_idx, params["mean"], params["covariance"])
        
        return hmm
    
    def save_training_results(self, output_dir: Path, trained_parameters: Dict[int, Dict[str, np.ndarray]]):
        """Save trained HMM parameters and statistics to disk.
        
        Args:
            output_dir: Directory to save results.
            trained_parameters: Trained parameters to save.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trained parameters
        params_file = output_dir / "hmm_parameters.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(trained_parameters, f)
        
        # Save training statistics
        stats_file = output_dir / "training_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.training_statistics, f, indent=2)
        
        # Save training metadata
        metadata = {
            "dataset_split": self.dataset.split,
            "feature_dimension": self.feature_dimension,
            "num_visemes_trained": len(trained_parameters),
            "total_training_samples": sum(params["num_samples"] for params in trained_parameters.values()),
            "viseme_sample_counts": {str(vid): params["num_samples"] 
                                   for vid, params in trained_parameters.items()}
        }
        
        metadata_file = output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training results saved to {output_dir}")
        print(f"  Parameters: {params_file}")
        print(f"  Statistics: {stats_file}")
        print(f"  Metadata: {metadata_file}")
    
    @staticmethod
    def load_training_results(model_dir: Path) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, float]]:
        """Load trained HMM parameters and statistics from disk.
        
        Args:
            model_dir: Directory containing saved model.
            
        Returns:
            Tuple of (trained_parameters, training_statistics).
        """
        model_dir = Path(model_dir)
        
        # Load parameters
        params_file = model_dir / "hmm_parameters.pkl"
        with open(params_file, 'rb') as f:
            trained_parameters = pickle.load(f)
        
        # Load statistics
        stats_file = model_dir / "training_statistics.json"
        with open(stats_file, 'r') as f:
            training_statistics = json.load(f)
        
        print(f"Loaded training results from {model_dir}")
        print(f"  {len(trained_parameters)} visemes with parameters")
        print(f"  Training statistics: μ={training_statistics['mu']:.4f}, σ={training_statistics['sigma']:.4f}")
        
        return trained_parameters, training_statistics


class TrainedHMMBuilder:
    """Builder for creating HMMs with pre-trained parameters."""
    
    def __init__(self, trained_parameters: Dict[int, Dict[str, np.ndarray]], 
                 training_statistics: Dict[str, float]):
        """Initialize with trained parameters.
        
        Args:
            trained_parameters: Trained HMM emission parameters.
            training_statistics: Training dataset statistics.
        """
        self.trained_parameters = trained_parameters
        self.training_statistics = training_statistics
        
        # Determine feature dimension from parameters
        if trained_parameters:
            first_params = next(iter(trained_parameters.values()))
            self.feature_dimension = first_params["mean"].shape[0]
        else:
            raise ValueError("No trained parameters provided")
    
    def build_hmm(self, viseme_sequence: List[int]) -> GaussianHMM:
        """Build HMM with trained parameters for given viseme sequence.
        
        Args:
            viseme_sequence: Target viseme sequence.
            
        Returns:
            GaussianHMM with trained emission parameters.
        """
        hmm = GaussianHMM(len(viseme_sequence), self.feature_dimension)
        
        for state_idx, viseme_id in enumerate(viseme_sequence):
            if viseme_id in self.trained_parameters:
                params = self.trained_parameters[viseme_id]
                hmm.set_emission_params(state_idx, params["mean"], params["covariance"])
            # If viseme not in training data, HMM uses default parameters
        
        return hmm
    
    def get_reference_baseline(self) -> ReferenceBaseline:
        """Create ReferenceBaseline with training statistics.
        
        Returns:
            ReferenceBaseline configured with training statistics.
        """
        baseline = ReferenceBaseline()
        
        # Set default statistics from training
        baseline.default_statistics = {
            "mu": self.training_statistics["mu"],
            "sigma": self.training_statistics["sigma"]
        }
        
        return baseline