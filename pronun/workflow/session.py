"""Practice session loop — orchestrates recording, scoring, and feedback."""

import tempfile
import os
import sys

_DEFAULT_EMISSIONS_PATH = "models/hmm_emissions.npz"
_DEFAULT_BASELINE_PATH = "models/reference_baseline.npz"
from typing import Optional

import numpy as np

from pronun.audio.g2p import text_to_ipa, text_to_arpabet, text_to_ipa_by_word
from pronun.audio.gop_scorer import compute_gop, overall_gop_score
from pronun.audio.phoneme_recognizer import recognize
from pronun.scoring.combiner import adaptive_combine, combine_scores
from pronun.scoring.feedback import (
    generate_feedback,
    generate_word_feedback,
    overall_feedback,
)
from pronun.visual.features.feature_builder import build_feature_sequence
from pronun.visual.features.landmark_extractor import LandmarkExtractor
from pronun.visual.features.normalizer import normalize_sequence
from pronun.visual.features.baseline_recorder import BaselineRecorder, ExponentialMovingAverageFilter
from pronun.visual.scoring.reference import ReferenceBaseline, UniversalReferenceBaseline
from pronun.visual.scoring.visual_scorer import VisualScorer
from pronun.training.train_hmm_emissions import load_trained_emissions
from pronun.visual.viseme.kmeans_viseme import KMeansViseme
from pronun.visual.viseme.lee_viseme import LeeViseme
from pronun.workflow.camera import Camera
from pronun.workflow.recorder_sync import SyncRecorder
from pronun.workflow.tracker import SessionTracker


class Session:
    """A pronunciation practice session."""

    def __init__(
        self,
        use_camera: bool = True,
        mode: str = "both",  # "A", "B", or "both"
        kmeans_model: KMeansViseme | None = None,
        enable_baseline_recording: bool = False,  # Disabled for speaker-independent system
        ema_alpha: float = 0.15,
        hmm_emissions_path: str | None = None,  # Path to trained HMM emissions
        reference_baseline_path: str | None = None  # Path to calibrated baseline
    ):
        self.use_camera = use_camera
        self.mode = mode.upper()
        self.kmeans_model = kmeans_model
        self.enable_baseline_recording = enable_baseline_recording
        self.ema_alpha = ema_alpha
        
        self.lee_viseme = LeeViseme()
        self.landmark_extractor = None
        self.camera = None
        self.tracker = SessionTracker()
        
        # Auto-detect default model paths if not explicitly provided
        if hmm_emissions_path is None and os.path.exists(_DEFAULT_EMISSIONS_PATH):
            hmm_emissions_path = _DEFAULT_EMISSIONS_PATH
        if reference_baseline_path is None and os.path.exists(_DEFAULT_BASELINE_PATH):
            reference_baseline_path = _DEFAULT_BASELINE_PATH

        # Load trained parameters if provided
        self.trained_emissions = None
        if hmm_emissions_path:
            try:
                self.trained_emissions = load_trained_emissions(hmm_emissions_path)
                print(f"Loaded trained HMM emissions for {len(self.trained_emissions)} visemes")
            except Exception as e:
                print(f"Warning: Failed to load HMM emissions from {hmm_emissions_path}: {e}")
        
        # Load calibrated reference baseline if provided
        reference_baseline = None
        if reference_baseline_path:
            try:
                reference_baseline = UniversalReferenceBaseline()
                reference_baseline.load(reference_baseline_path)
                stats = reference_baseline.get_universal_statistics()
                print(f"Loaded calibrated reference baseline: μ_ref={stats['mu']:.3f}, σ_ref={stats['sigma']:.3f}")
            except Exception as e:
                print(f"Warning: Failed to load reference baseline from {reference_baseline_path}: {e}")
                reference_baseline = None
        
        # Initialize visual scorer with trained components
        self.visual_scorer = VisualScorer(reference=reference_baseline)
        
        # EMA temporal smoothing for speaker-independent system
        self.ema_filter: Optional[ExponentialMovingAverageFilter] = None
        # Legacy baseline recorder (disabled by default)
        self.baseline_recorder: Optional[BaselineRecorder] = None
        self.baseline_ready = False

    def setup(self):
        """Initialize camera and landmark extractor if using video."""
        if self.use_camera:
            try:
                self.camera = Camera()
                self.camera.open()
                self.landmark_extractor = LandmarkExtractor()
                
                # Initialize EMA temporal smoothing for speaker-independent system
                self.ema_filter = ExponentialMovingAverageFilter(alpha=self.ema_alpha)
                
                # Optional baseline recorder (disabled by default for speaker-independent system)
                if self.enable_baseline_recording:
                    self.baseline_recorder = BaselineRecorder(
                        self.camera, self.landmark_extractor
                    )
                
            except (RuntimeError, AttributeError, ImportError, OSError) as e:
                self.use_camera = False
                self.camera = None
                print(
                    f"\n[Warning] Visual scoring disabled: {e}\n"
                    "[Warning] Possible causes:\n"
                    "  - Camera permission not granted "
                    "(System Settings → Privacy & Security → Camera)\n"
                    "  - MediaPipe model download failed (SSL issue) — "
                    "check ~/.cache/pronun/face_landmarker.task exists\n"
                    "[Warning] Falling back to audio-only scoring (Visual: N/A)\n",
                    file=sys.stderr,
                )

    def teardown(self):
        """Release resources."""
        if self.camera:
            self.camera.close()
        if self.landmark_extractor:
            self.landmark_extractor.close()

    def _record_and_get_audio_visual(self):
        """Record audio+video and return (audio, video_frames, audio_path)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        recorder = SyncRecorder(camera=self.camera)
        audio, video_frames = recorder.record(audio_path)
        return audio, video_frames, audio_path

    def record_baseline_mouth_state(self, duration_seconds: float = 1.5) -> dict:
        """Record baseline mouth state for adaptive normalization (Step 1 of online usage).
        
        User should maintain neutral mouth position during recording.
        
        Args:
            duration_seconds: Duration to record baseline (1-2 seconds recommended).
            
        Returns:
            Dict with baseline recording results.
        """
        if not self.baseline_recorder:
            return {
                "success": False,
                "error": "Baseline recording not enabled or camera not available"
            }
        
        result = self.baseline_recorder.record_baseline(duration_seconds)
        if result.get("success", False):
            self.baseline_ready = True
        
        return result
    
    def get_baseline_info(self) -> dict:
        """Get information about the recorded baseline."""
        if self.baseline_recorder:
            return self.baseline_recorder.get_baseline_info()
        return {"baseline_ready": False, "error": "Baseline recorder not available"}

    def _compute_visual_score(self, video_frames, text):
        """Compute visual scores from video frames. Returns detailed results."""
        result = {
            "visual_score_a": None,
            "visual_score_b": None,
            "visual_score": None,
            "visual_details_a": None,
            "visual_details_b": None,
        }

        if not (self.use_camera and video_frames and self.landmark_extractor):
            return result

        landmarks = self.landmark_extractor.extract_sequence(video_frames)
        normalized = normalize_sequence(landmarks)
        features = build_feature_sequence(normalized)

        if not features:
            return result

        # Apply adaptive normalization if baseline is available (legacy, disabled by default)
        if self.baseline_recorder and self.baseline_recorder.is_baseline_ready():
            try:
                features = self.baseline_recorder.apply_adaptive_normalization(features)
            except RuntimeError as e:
                print(f"Warning: Adaptive normalization failed: {e}")

        # Apply EMA temporal smoothing (speaker-independent)
        if self.ema_filter:
            self.ema_filter.reset_filter()  # Reset for new sequence
            smoothed_features = []
            for feature in features:
                smoothed_feature = self.ema_filter.apply_filter(feature)
                smoothed_features.append(smoothed_feature)
            features = smoothed_features

        obs = np.array(features)

        if self.mode in ("B", "BOTH"):
            viseme_seq = self.lee_viseme.text_to_viseme_sequence(text)
            
            # Build HMM (initially untrained)
            hmm = self.visual_scorer.build_hmm(viseme_seq, {}, obs.shape[1])
            
            # Apply trained emission parameters if available
            if self.trained_emissions:
                hmm = self._apply_trained_emissions_to_hmm(hmm, viseme_seq)
            
            score_b = self.visual_scorer.score(hmm, obs, viseme_seq)
            result["visual_score_b"] = score_b["score"]
            result["visual_details_b"] = score_b

        if self.mode in ("A", "BOTH") and self.kmeans_model:
            pred_visemes = self.kmeans_model.predict(obs)
            pred_viseme_list = list(pred_visemes)
            
            # Build HMM (initially untrained)
            hmm = self.visual_scorer.build_hmm(pred_viseme_list, {}, obs.shape[1])
            
            # Apply trained emission parameters if available
            if self.trained_emissions:
                hmm = self._apply_trained_emissions_to_hmm(hmm, pred_viseme_list)
            
            score_a = self.visual_scorer.score(hmm, obs, pred_viseme_list)
            result["visual_score_a"] = score_a["score"]
            result["visual_details_a"] = score_a

        # Choose primary visual score (prefer Mode B)
        if result["visual_score_b"] is not None:
            result["visual_score"] = result["visual_score_b"]
        elif result["visual_score_a"] is not None:
            result["visual_score"] = result["visual_score_a"]

        return result

    def _apply_trained_emissions_to_hmm(self, hmm: 'GaussianHMM', viseme_sequence: list[int]) -> 'GaussianHMM':
        """Apply trained emission parameters to HMM states.
        
        Args:
            hmm: HMM to modify (created with build_hmm).
            viseme_sequence: Sequence of viseme IDs corresponding to HMM states.
            
        Returns:
            Modified HMM with trained parameters.
        """
        if not self.trained_emissions:
            return hmm
            
        for state_idx, viseme_id in enumerate(viseme_sequence):
            if viseme_id in self.trained_emissions:
                params = self.trained_emissions[viseme_id]
                hmm.set_emission_params(state_idx, params['mean'], params['variance'])
                
        return hmm

    def practice_word(self, word: str) -> dict:
        """Run a single word practice iteration.

        Args:
            word: The word to practice.

        Returns:
            Dict with all scoring results and feedback.
        """
        target_ipa = text_to_ipa(word)
        target_arpabet = text_to_arpabet(word)

        audio, video_frames, audio_path = self._record_and_get_audio_visual()

        try:
            # Audio scoring
            result = recognize(audio)
            phoneme_scores = compute_gop(
                log_probs=result.log_probs,
                predicted_ids=result.predicted_ids,
                predicted_phonemes=result.predicted_phonemes,
                target_phonemes=target_ipa,
                vocab=result.vocab,
            )
            audio_score = overall_gop_score(phoneme_scores)

            # Visual scoring
            visual_results = self._compute_visual_score(video_frames, word)
            visual_score_a = visual_results["visual_score_a"]
            visual_score_b = visual_results["visual_score_b"]
            visual_score = visual_results["visual_score"]

            # Combined scoring
            combined = combine_scores(audio_score, visual_score)
            per_phoneme = adaptive_combine(phoneme_scores, visual_score)
            fb = generate_feedback(per_phoneme)
            overall_msg = overall_feedback(combined)

            return {
                "word": word,
                "target_phonemes": target_ipa,
                "audio_score": audio_score,
                "visual_score_a": visual_score_a,
                "visual_score_b": visual_score_b,
                "visual_score": visual_score,
                "visual_details_a": visual_results["visual_details_a"],
                "visual_details_b": visual_results["visual_details_b"],
                "combined_score": combined,
                "phoneme_details": per_phoneme,
                "feedback": fb,
                "overall_feedback": overall_msg,
            }
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def practice_sentence(self, sentence: str) -> dict:
        """Run a full sentence practice iteration.

        Records audio+video ONCE for the entire sentence, runs G2P with
        word segmentation, scores phonemes, and maps scores back to words.

        Args:
            sentence: The sentence to practice.

        Returns:
            Dict with sentence_score, word_scores, phoneme_details, feedback.
        """
        target_ipa, word_segments = text_to_ipa_by_word(sentence)

        audio, video_frames, audio_path = self._record_and_get_audio_visual()

        try:
            # Audio scoring on full sentence
            result = recognize(audio)
            phoneme_scores = compute_gop(
                log_probs=result.log_probs,
                predicted_ids=result.predicted_ids,
                predicted_phonemes=result.predicted_phonemes,
                target_phonemes=target_ipa,
                vocab=result.vocab,
            )
            audio_score = overall_gop_score(phoneme_scores)

            # Visual scoring
            visual_results = self._compute_visual_score(video_frames, sentence)
            visual_score_a = visual_results["visual_score_a"]
            visual_score_b = visual_results["visual_score_b"]
            visual_score = visual_results["visual_score"]

            # Combined per-phoneme scoring
            combined = combine_scores(audio_score, visual_score)
            per_phoneme = adaptive_combine(phoneme_scores, visual_score)

            # Map phoneme scores to words using segment boundaries
            word_scores = []
            for seg in word_segments:
                seg_phonemes = per_phoneme[seg["start"]:seg["end"]]
                if seg_phonemes:
                    word_avg = sum(
                        p.get("combined_score", p.get("gop_score", 0))
                        for p in seg_phonemes
                    ) / len(seg_phonemes)
                else:
                    word_avg = 0
                word_scores.append({
                    "word": seg["word"],
                    "score": word_avg,
                    "phoneme_start": seg["start"],
                    "phoneme_end": seg["end"],
                })

            # Sentence score is the average of word scores
            sentence_score = (
                sum(ws["score"] for ws in word_scores) / len(word_scores)
                if word_scores else combined
            )

            # Feedback
            word_fb = generate_word_feedback(word_scores)
            phoneme_fb = generate_feedback(per_phoneme)
            overall_msg = overall_feedback(sentence_score, word_scores)

            practice_result = {
                "sentence": sentence,
                "sentence_score": sentence_score,
                "audio_score": audio_score,
                "visual_score_a": visual_score_a,
                "visual_score_b": visual_score_b,
                "visual_score": visual_score,
                "visual_details_a": visual_results["visual_details_a"],
                "visual_details_b": visual_results["visual_details_b"],
                "combined_score": combined,
                "word_scores": word_fb,
                "word_segments": word_segments,
                "phoneme_details": per_phoneme,
                "feedback": phoneme_fb,
                "overall_feedback": overall_msg,
            }

            self.tracker.record(practice_result)
            return practice_result

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def get_progress(self, sentence: str = None) -> dict:
        """Get session progress data from the tracker."""
        return {
            "history": self.tracker.get_history(sentence),
            "trend": self.tracker.get_trend(sentence),
            "summary": self.tracker.summary(),
        }

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.teardown()
