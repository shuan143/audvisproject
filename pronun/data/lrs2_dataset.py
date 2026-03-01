"""LRS2 Dataset loader for statistical visual speech modeling."""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import numpy as np
import cv2

from pronun.audio.g2p import text_to_arpabet
from pronun.visual.viseme.lee_viseme import LeeViseme


class LRS2Dataset:
    """LRS2 Dataset handler for robust statistical visual speech modeling.
    
    Handles LRS2's natural sentence audiovisual sequences with transcript alignment.
    Supports pretrain/train/val/test splits for comprehensive statistical modeling.
    """
    
    def __init__(self, corpus_path: str, split: str = "train"):
        """Initialize LRS2 dataset.
        
        Args:
            corpus_path: Path to LRS2 root directory.
            split: Dataset split ('pretrain', 'train', 'val', 'test').
        """
        self.corpus_path = Path(corpus_path)
        self.split = split.lower()
        self.lee_viseme = LeeViseme()
        
        # LRS2 structure: {split}/{speaker_id}/{video_id}.mp4 and {video_id}.txt
        self._video_files = []
        self._transcript_files = []
        self._speaker_ids = []
        
        # Statistics for robust training
        self._dataset_stats = {}
        
    def scan_corpus(self) -> Dict[str, any]:
        """Scan LRS2 dataset directory structure.
        
        Returns:
            Dict with comprehensive dataset statistics.
        """
        split_path = self.corpus_path / self.split
        if not split_path.exists():
            raise FileNotFoundError(f"LRS2 split '{self.split}' not found at {split_path}")
            
        self._video_files = []
        self._transcript_files = []
        self._speaker_ids = []
        
        # LRS2 directory structure scanning
        speakers = set()
        total_videos = 0
        
        for speaker_dir in split_path.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                speakers.add(speaker_id)
                
                # Find video and transcript pairs
                video_files = list(speaker_dir.glob("*.mp4"))
                for video_file in video_files:
                    transcript_file = video_file.with_suffix(".txt")
                    if transcript_file.exists():
                        self._video_files.append(video_file)
                        self._transcript_files.append(transcript_file)
                        self._speaker_ids.append(speaker_id)
                        total_videos += 1
        
        # Compute dataset statistics
        self._dataset_stats = {
            "split": self.split,
            "total_speakers": len(speakers),
            "total_videos": total_videos,
            "speakers": sorted(list(speakers)),
            "avg_videos_per_speaker": total_videos / len(speakers) if speakers else 0
        }
        
        return self._dataset_stats
    
    def get_samples(self) -> Iterator[Tuple[Path, str, List[int], str]]:
        """Get dataset samples as iterator for memory efficiency.
        
        Yields:
            Tuples of (video_path, transcript, viseme_sequence, speaker_id).
        """
        if not self._video_files:
            raise RuntimeError("Must call scan_corpus() first")
            
        for video_path, transcript_path, speaker_id in zip(
            self._video_files, self._transcript_files, self._speaker_ids
        ):
            transcript = self._load_transcript(transcript_path)
            if transcript:  # Skip empty transcripts
                viseme_seq = self._text_to_viseme_sequence(transcript)
                yield video_path, transcript, viseme_seq, speaker_id
    
    def get_sample_by_index(self, index: int) -> Tuple[Path, str, List[int], str]:
        """Get specific sample by index.
        
        Args:
            index: Sample index.
            
        Returns:
            Tuple of (video_path, transcript, viseme_sequence, speaker_id).
        """
        if not self._video_files or index >= len(self._video_files):
            raise IndexError(f"Sample index {index} out of range")
            
        video_path = self._video_files[index]
        transcript_path = self._transcript_files[index]
        speaker_id = self._speaker_ids[index]
        
        transcript = self._load_transcript(transcript_path)
        viseme_seq = self._text_to_viseme_sequence(transcript)
        
        return video_path, transcript, viseme_seq, speaker_id
    
    def get_speaker_samples(self, speaker_id: str) -> List[Tuple[Path, str, List[int]]]:
        """Get all samples for a specific speaker.
        
        Args:
            speaker_id: Target speaker identifier.
            
        Returns:
            List of (video_path, transcript, viseme_sequence) tuples for the speaker.
        """
        speaker_samples = []
        for video_path, transcript_path, spk_id in zip(
            self._video_files, self._transcript_files, self._speaker_ids
        ):
            if spk_id == speaker_id:
                transcript = self._load_transcript(transcript_path)
                if transcript:
                    viseme_seq = self._text_to_viseme_sequence(transcript)
                    speaker_samples.append((video_path, transcript, viseme_seq))
        
        return speaker_samples
    
    def _load_transcript(self, transcript_path: Path) -> str:
        """Load and clean transcript from LRS2 format.
        
        Args:
            transcript_path: Path to transcript file.
            
        Returns:
            Cleaned transcript text.
        """
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # LRS2 format: "Text: ACTUAL_TRANSCRIPT"
            transcript = ""
            for line in lines:
                if line.startswith("Text:"):
                    transcript = line[5:].strip()
                    break
            
            if not transcript:
                # Fallback: take first non-empty line
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        transcript = line
                        break
            
            # Clean transcript for natural speech processing
            transcript = self._clean_transcript(transcript)
            
            return transcript
            
        except Exception as e:
            print(f"Warning: Failed to load transcript {transcript_path}: {e}")
            return ""
    
    def _clean_transcript(self, transcript: str) -> str:
        """Clean transcript for robust natural speech processing.
        
        Args:
            transcript: Raw transcript text.
            
        Returns:
            Cleaned transcript suitable for phoneme processing.
        """
        if not transcript:
            return ""
            
        # Convert to lowercase
        transcript = transcript.lower()
        
        # Remove special LRS2 markers and noise
        transcript = re.sub(r'\{[^}]*\}', '', transcript)  # Remove {NOISE} markers
        transcript = re.sub(r'\[[^]]*\]', '', transcript)  # Remove [SPEAKER] markers
        transcript = re.sub(r'<[^>]*>', '', transcript)    # Remove <UNCLEAR> markers
        
        # Handle contractions and informal speech
        contractions = {
            "can't": "cannot",
            "won't": "will not", 
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            transcript = transcript.replace(contraction, expansion)
        
        # Remove extra punctuation and whitespace
        transcript = re.sub(r'[^\w\s]', ' ', transcript)
        transcript = re.sub(r'\s+', ' ', transcript)
        transcript = transcript.strip()
        
        return transcript
    
    def _text_to_viseme_sequence(self, text: str) -> List[int]:
        """Convert natural speech text to viseme sequence.
        
        Args:
            text: Cleaned natural speech transcript.
            
        Returns:
            List of viseme IDs using Lee's mapping.
        """
        if not text:
            return []
            
        try:
            return self.lee_viseme.text_to_viseme_sequence(text)
        except Exception as e:
            print(f"Warning: Failed to convert text to visemes '{text}': {e}")
            return []
    
    def compute_corpus_statistics(self) -> Dict[str, any]:
        """Compute comprehensive corpus statistics for training.
        
        Returns:
            Dict with statistical analysis of the corpus.
        """
        if not self._video_files:
            raise RuntimeError("Must call scan_corpus() first")
        
        # Analyze transcript and viseme statistics
        transcript_lengths = []
        viseme_counts = {}
        unique_visemes = set()
        word_counts = {}
        speaker_stats = {}
        
        print(f"Computing statistics for {len(self._video_files)} samples...")
        
        for i, (_, transcript_path, speaker_id) in enumerate(zip(
            self._video_files, self._transcript_files, self._speaker_ids
        )):
            if i % 1000 == 0:
                print(f"Processing sample {i+1}/{len(self._video_files)}")
                
            transcript = self._load_transcript(transcript_path)
            if not transcript:
                continue
                
            # Transcript analysis
            words = transcript.split()
            transcript_lengths.append(len(words))
            
            # Word frequency analysis
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Viseme analysis
            viseme_seq = self._text_to_viseme_sequence(transcript)
            unique_visemes.update(viseme_seq)
            
            for viseme_id in viseme_seq:
                viseme_counts[viseme_id] = viseme_counts.get(viseme_id, 0) + 1
            
            # Speaker statistics
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {"samples": 0, "total_words": 0}
            speaker_stats[speaker_id]["samples"] += 1
            speaker_stats[speaker_id]["total_words"] += len(words)
        
        # Compile comprehensive statistics
        statistics = {
            "dataset_info": self._dataset_stats,
            "transcript_statistics": {
                "total_transcripts": len([t for t in transcript_lengths if t > 0]),
                "avg_transcript_length": np.mean(transcript_lengths) if transcript_lengths else 0,
                "transcript_length_std": np.std(transcript_lengths) if transcript_lengths else 0,
                "min_length": min(transcript_lengths) if transcript_lengths else 0,
                "max_length": max(transcript_lengths) if transcript_lengths else 0,
                "total_unique_words": len(word_counts),
                "total_word_tokens": sum(word_counts.values())
            },
            "viseme_statistics": {
                "total_viseme_tokens": sum(viseme_counts.values()),
                "unique_visemes": len(unique_visemes),
                "viseme_distribution": viseme_counts,
                "most_common_visemes": sorted(viseme_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]
            },
            "speaker_statistics": {
                "total_speakers": len(speaker_stats),
                "avg_samples_per_speaker": np.mean([s["samples"] for s in speaker_stats.values()]),
                "avg_words_per_speaker": np.mean([s["total_words"] for s in speaker_stats.values()]),
                "speaker_sample_distribution": {k: v["samples"] for k, v in speaker_stats.items()}
            }
        }
        
        return statistics
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self._video_files)
    
    def get_dataset_info(self) -> Dict[str, any]:
        """Get basic dataset information."""
        return self._dataset_stats


class LRS2VideoProcessor:
    """Efficient video processing for LRS2 dataset."""
    
    @staticmethod
    def load_video_frames(video_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Load video frames from LRS2 mp4 file.
        
        Args:
            video_path: Path to video file.
            max_frames: Maximum number of frames to load (None for all).
            
        Returns:
            List of video frames as numpy arrays.
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video {video_path}")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB for MediaPipe compatibility
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
            
            cap.release()
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return []
        
        return frames
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, any]:
        """Get video metadata.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Dict with video information.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"error": "Cannot open video"}
            
            info = {
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            return {"error": str(e)}