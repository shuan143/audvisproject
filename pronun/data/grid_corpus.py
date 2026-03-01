"""GRID Corpus dataset preparation for visual speech modeling."""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from pronun.audio.g2p import text_to_arpabet
from pronun.visual.viseme.lee_viseme import LeeViseme


class GridCorpusDataset:
    """GRID Corpus dataset handler for visual speech modeling.
    
    Handles multi-speaker audiovisual speech sequences with transcript alignment.
    Provides train/validation splits for statistical visual speech modeling.
    """
    
    def __init__(self, corpus_path: str, train_split: float = 0.8, 
                 max_videos: int = None, max_speakers: int = None):
        """Initialize GRID Corpus dataset.
        
        Args:
            corpus_path: Path to GRID Corpus root directory.
            train_split: Fraction of data for training (rest for validation).
            max_videos: Maximum number of videos to use (None = all).
            max_speakers: Maximum number of speakers to use (None = all).
        """
        self.corpus_path = Path(corpus_path)
        self.train_split = train_split
        self.max_videos = max_videos
        self.max_speakers = max_speakers
        self.lee_viseme = LeeViseme()
        
        # GRID Corpus structure: s{speaker_id}/{video_file}.mpg
        # Transcripts: alignments/s{speaker_id}/{video_file}.align
        self._video_files = []
        self._transcript_files = []
        self._train_files = []
        self._val_files = []
        
    def scan_corpus(self) -> Dict[str, int]:
        """Scan GRID Corpus directory structure.
        
        Returns:
            Dict with corpus statistics.
        """
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"GRID Corpus not found at {self.corpus_path}")
            
        self._video_files = []
        self._transcript_files = []
        
        # Scan speaker directories (only directories, not zip files)
        speakers = []
        for speaker_dir in sorted(self.corpus_path.iterdir()):
            if (speaker_dir.is_dir() and 
                speaker_dir.name.startswith('s') and 
                speaker_dir.name[1:].isdigit()):  # Ensure it's s + digits only
                speakers.append(speaker_dir.name)
                
                # Find video and transcript pairs  
                video_files = list(speaker_dir.glob("*.mpg"))
                alignments_dir = self.corpus_path / "alignments" / speaker_dir.name
                
                for video_file in video_files:
                    # Look for corresponding .align file in alignments directory
                    align_file = alignments_dir / video_file.with_suffix(".align").name
                    if align_file.exists():
                        self._video_files.append(video_file)
                        self._transcript_files.append(align_file)
                        
                        # Apply video limit
                        if self.max_videos and len(self._video_files) >= self.max_videos:
                            break
                
                # Apply speaker limit after processing videos
                if self.max_speakers and len(speakers) >= self.max_speakers:
                    break
                    
                if self.max_videos and len(self._video_files) >= self.max_videos:
                    break
        
        # Create train/validation split
        indices = list(range(len(self._video_files)))
        random.shuffle(indices)
        split_idx = int(len(indices) * self.train_split)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        self._train_files = [(self._video_files[i], self._transcript_files[i]) 
                           for i in train_indices]
        self._val_files = [(self._video_files[i], self._transcript_files[i]) 
                         for i in val_indices]
        
        return {
            "total_speakers": len(speakers),
            "total_videos": len(self._video_files),
            "train_videos": len(self._train_files),
            "val_videos": len(self._val_files),
            "speakers": speakers
        }
    
    def get_train_samples(self) -> List[Tuple[Path, str, List[int]]]:
        """Get training samples with video path, transcript, and viseme sequence.
        
        Returns:
            List of (video_path, transcript, viseme_sequence) tuples.
        """
        samples = []
        for video_path, transcript_path in self._train_files:
            transcript = self._load_transcript(transcript_path)
            viseme_seq = self._text_to_viseme_sequence(transcript)
            samples.append((video_path, transcript, viseme_seq))
        return samples
    
    def get_validation_samples(self) -> List[Tuple[Path, str, List[int]]]:
        """Get validation samples with video path, transcript, and viseme sequence.
        
        Returns:
            List of (video_path, transcript, viseme_sequence) tuples.
        """
        samples = []
        for video_path, transcript_path in self._val_files:
            transcript = self._load_transcript(transcript_path)
            viseme_seq = self._text_to_viseme_sequence(transcript)
            samples.append((video_path, transcript, viseme_seq))
        return samples
    
    def _load_transcript(self, transcript_path: Path) -> str:
        """Load transcript from alignment file.
        
        Args:
            transcript_path: Path to .align file.
            
        Returns:
            Cleaned transcript text.
        """
        words = []
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: start_time end_time word
                    parts = line.split()
                    if len(parts) >= 3:
                        word = parts[2].lower()
                        # Skip silence markers
                        if word != 'sil':
                            words.append(word)
        
        # Join words into transcript
        transcript = ' '.join(words)
        return transcript
    
    def _text_to_viseme_sequence(self, text: str) -> List[int]:
        """Convert text to viseme sequence using Lee's mapping.
        
        Args:
            text: Input transcript text.
            
        Returns:
            List of viseme IDs.
        """
        return self.lee_viseme.text_to_viseme_sequence(text)
    
    def get_corpus_statistics(self) -> Dict[str, any]:
        """Compute dataset statistics for training parameter estimation.
        
        Returns:
            Dict with corpus-wide statistics.
        """
        if not self._train_files:
            raise RuntimeError("Must call scan_corpus() first")
        
        viseme_counts = {}
        transcript_lengths = []
        unique_visemes = set()
        
        for _, transcript_path in self._train_files:
            transcript = self._load_transcript(transcript_path)
            viseme_seq = self._text_to_viseme_sequence(transcript)
            
            transcript_lengths.append(len(transcript.split()))
            unique_visemes.update(viseme_seq)
            
            for viseme_id in viseme_seq:
                viseme_counts[viseme_id] = viseme_counts.get(viseme_id, 0) + 1
        
        return {
            "total_viseme_tokens": sum(viseme_counts.values()),
            "unique_visemes": len(unique_visemes),
            "viseme_distribution": viseme_counts,
            "avg_transcript_length": np.mean(transcript_lengths),
            "transcript_length_std": np.std(transcript_lengths),
            "most_common_visemes": sorted(viseme_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        }


class GridCorpusFeatureExtractor:
    """Extract visual features from GRID Corpus videos for HMM training."""
    
    def __init__(self, dataset: GridCorpusDataset):
        """Initialize feature extractor.
        
        Args:
            dataset: GridCorpusDataset instance.
        """
        self.dataset = dataset
        
        # Import here to avoid circular dependencies
        from pronun.visual.features.landmark_extractor import LandmarkExtractor
        from pronun.visual.features.normalizer import normalize_sequence
        from pronun.visual.features.feature_builder import build_feature_sequence
        from pronun.visual.features.baseline_recorder import ExponentialMovingAverageFilter
        
        self.landmark_extractor = LandmarkExtractor()
        self.ema_filter = ExponentialMovingAverageFilter(alpha=0.15)
    
    def extract_training_features(self) -> Dict[int, np.ndarray]:
        """Extract features from training videos for HMM training.
        
        Process GRID corpus videos with complete pipeline:
        1. MediaPipe landmark extraction
        2. Normalization (centroid + width scaling)
        3. Feature building (geometric + temporal + velocity)
        4. EMA temporal smoothing
        5. Group by viseme ID for MLE training
        
        Returns:
            Dict mapping viseme_id -> training observations array.
        """
        from pronun.visual.features.normalizer import normalize_sequence
        from pronun.visual.features.feature_builder import build_feature_sequence
        import cv2
        
        viseme_features = {}  # viseme_id -> list of feature vectors
        
        train_samples = self.dataset.get_train_samples()
        print(f"Processing {len(train_samples)} training videos...")
        
        for video_idx, (video_path, transcript, viseme_seq) in enumerate(train_samples):
            if video_idx % 50 == 0:
                print(f"Processing video {video_idx + 1}/{len(train_samples)}: {video_path.name}")
            
            try:
                # Extract features from video
                features_per_viseme = self._process_video_with_alignment(
                    video_path, viseme_seq
                )
                
                # Accumulate features by viseme ID
                for viseme_id, features in features_per_viseme.items():
                    if viseme_id not in viseme_features:
                        viseme_features[viseme_id] = []
                    viseme_features[viseme_id].extend(features)
                    
            except Exception as e:
                print(f"Warning: Failed to process {video_path}: {e}")
                continue
        
        # Convert lists to arrays for MLE training
        viseme_observations = {}
        for viseme_id, feature_list in viseme_features.items():
            if len(feature_list) >= 2:  # Need at least 2 samples for variance
                viseme_observations[viseme_id] = np.array(feature_list)
                print(f"Viseme {viseme_id}: {len(feature_list)} features")
        
        self.landmark_extractor.close()
        return viseme_observations
    
    def _process_video_with_alignment(self, video_path: Path, 
                                     viseme_sequence: List[int]) -> Dict[int, List[np.ndarray]]:
        """Process single video and extract features per viseme with temporal alignment.
        
        Args:
            video_path: Path to video file.
            viseme_sequence: Expected viseme sequence for alignment.
            
        Returns:
            Dict mapping viseme_id -> list of feature vectors.
        """
        from pronun.visual.features.normalizer import normalize_sequence
        from pronun.visual.features.feature_builder import build_feature_sequence
        import cv2
        
        # Read video frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) == 0:
            return {}
        
        # Extract landmarks from all frames
        landmarks = self.landmark_extractor.extract_sequence(frames)
        
        # Normalize landmarks
        normalized = normalize_sequence(landmarks)
        
        # Build feature sequence
        features = build_feature_sequence(normalized)
        
        if len(features) == 0:
            return {}
        
        # Apply EMA temporal smoothing
        self.ema_filter.reset_filter()  # Reset for new video
        smoothed_features = []
        for feature in features:
            smoothed = self.ema_filter.apply_filter(feature)
            smoothed_features.append(smoothed)
        
        # Align features with viseme sequence using uniform distribution
        # Simple uniform alignment: divide features equally across visemes
        viseme_features = {}
        
        if len(viseme_sequence) > 0 and len(smoothed_features) > 0:
            frames_per_viseme = max(1, len(smoothed_features) // len(viseme_sequence))
            
            for i, viseme_id in enumerate(viseme_sequence):
                start_frame = i * frames_per_viseme
                end_frame = min((i + 1) * frames_per_viseme, len(smoothed_features))
                
                if start_frame < len(smoothed_features):
                    viseme_frames = smoothed_features[start_frame:end_frame]
                    
                    if viseme_id not in viseme_features:
                        viseme_features[viseme_id] = []
                    viseme_features[viseme_id].extend(viseme_frames)
        
        return viseme_features
    
    def extract_validation_features(self) -> List[Tuple[List[np.ndarray], List[int]]]:
        """Extract features from validation videos for reference baseline calibration.
        
        Returns:
            List of (feature_sequences, viseme_sequences) for validation.
        """
        from pronun.visual.features.normalizer import normalize_sequence
        from pronun.visual.features.feature_builder import build_feature_sequence
        import cv2
        
        validation_data = []
        val_samples = self.dataset.get_validation_samples()
        print(f"Processing {len(val_samples)} validation videos for calibration...")
        
        for video_idx, (video_path, transcript, viseme_seq) in enumerate(val_samples):
            if video_idx % 20 == 0:
                print(f"Processing validation video {video_idx + 1}/{len(val_samples)}")
            
            try:
                # Read video frames
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                cap.release()
                
                if len(frames) == 0:
                    continue
                
                # Extract and process features
                landmarks = self.landmark_extractor.extract_sequence(frames)
                normalized = normalize_sequence(landmarks)
                features = build_feature_sequence(normalized)
                
                if len(features) > 0:
                    # Apply EMA smoothing
                    self.ema_filter.reset_filter()
                    smoothed_features = []
                    for feature in features:
                        smoothed = self.ema_filter.apply_filter(feature)
                        smoothed_features.append(smoothed)
                    
                    validation_data.append((smoothed_features, viseme_seq))
                    
            except Exception as e:
                print(f"Warning: Failed to process validation video {video_path}: {e}")
                continue
        
        return validation_data