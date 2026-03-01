#!/usr/bin/env python3
"""Calibrate reference baseline from GRID Corpus validation data.

This script implements Task #3 from CLAUDE.md:
- Load trained HMM emission parameters
- Run Forward Algorithm on validation speakers
- Compute universal reference statistics (μ_ref, σ_ref)
- Save calibrated baseline for Session loading

Usage:
    python -m pronun.training.calibrate_reference /path/to/grid/corpus --emissions models/hmm_emissions.npz --output models/reference_baseline.npz
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

from pronun.data.grid_corpus import GridCorpusDataset, GridCorpusFeatureExtractor
from pronun.visual.scoring.hmm import GaussianHMM
from pronun.visual.scoring.reference import UniversalReferenceBaseline
from pronun.visual.scoring.visual_scorer import VisualScorer
from pronun.training.train_hmm_emissions import load_trained_emissions
from pronun.visual.features.feature_builder import feature_dim
from pronun.config import ALL_LIP_INDICES


def build_trained_hmm(viseme_sequence: List[int], trained_emissions: Dict[int, Dict]) -> GaussianHMM:
    """Build HMM with trained emission parameters for a viseme sequence.
    
    Args:
        viseme_sequence: List of viseme IDs.
        trained_emissions: Dict of trained emission parameters.
        
    Returns:
        GaussianHMM with trained parameters.
    """
    if not viseme_sequence:
        return None
    
    feature_dimension = feature_dim(len(ALL_LIP_INDICES))
    hmm = GaussianHMM(num_states=len(viseme_sequence), feature_dim=feature_dimension)
    
    # Set trained emission parameters for each state
    for state, viseme_id in enumerate(viseme_sequence):
        if viseme_id in trained_emissions:
            params = trained_emissions[viseme_id]
            hmm.set_emission_params(state, params['mean'], params['variance'])
        else:
            # Fallback to default parameters if viseme not trained
            mean = np.zeros(feature_dimension)
            variance = np.ones(feature_dimension)
            hmm.set_emission_params(state, mean, variance)
            print(f"Warning: No trained parameters for viseme {viseme_id}, using defaults")
    
    return hmm


def calibrate_reference_baseline(corpus_path: str, emissions_path: str, output_path: str, 
                               train_split: float = 0.8, max_videos: int = None, 
                               max_speakers: int = None) -> Dict[str, any]:
    """Calibrate universal reference baseline from validation data.
    
    Args:
        corpus_path: Path to GRID corpus root directory.
        emissions_path: Path to trained emission parameters.
        output_path: Path to save calibrated baseline.
        train_split: Fraction for training (rest used for calibration).
        max_videos: Maximum number of videos to use (None = all).
        max_speakers: Maximum number of speakers to use (None = all).
        
    Returns:
        Dict with calibration results.
    """
    print("=== Reference Baseline Calibration ===")
    print(f"Corpus path: {corpus_path}")
    print(f"Emissions path: {emissions_path}")
    print(f"Output path: {output_path}")
    print()
    
    # Load trained emission parameters
    print("Loading trained HMM emission parameters...")
    trained_emissions = load_trained_emissions(emissions_path)
    print(f"Loaded parameters for {len(trained_emissions)} visemes")
    print()
    
    # Initialize dataset with limits  
    dataset = GridCorpusDataset(corpus_path, train_split=train_split,
                               max_videos=max_videos, max_speakers=max_speakers)
    
    if max_videos:
        print(f"Limited to {max_videos} videos max")
    if max_speakers:
        print(f"Limited to {max_speakers} speakers max")
    
    # Scan corpus structure
    print("Scanning GRID corpus...")
    corpus_stats = dataset.scan_corpus()
    print(f"Using {corpus_stats['val_videos']} validation videos for calibration")
    print()
    
    # Extract validation features
    feature_extractor = GridCorpusFeatureExtractor(dataset)
    print("Extracting validation features...")
    validation_data = feature_extractor.extract_validation_features()
    print(f"Processed {len(validation_data)} validation sequences")
    print()
    
    # Run Forward Algorithm on validation data with trained HMMs
    print("Computing log-likelihoods with trained HMMs...")
    log_likelihoods_norm = []
    processed_sequences = 0
    
    for features, viseme_sequence in validation_data:
        if len(features) == 0 or len(viseme_sequence) == 0:
            continue
        
        try:
            # Build HMM with trained parameters
            hmm = build_trained_hmm(viseme_sequence, trained_emissions)
            if hmm is None:
                continue
            
            # Convert features to numpy array
            feature_array = np.array(features)
            
            # Run Forward Algorithm
            log_likelihood = hmm.forward(feature_array)
            
            if np.isfinite(log_likelihood):
                # Normalize by sequence length: L_norm = L / T
                log_likelihood_norm = log_likelihood / len(features)
                log_likelihoods_norm.append(log_likelihood_norm)
                processed_sequences += 1
                
                if processed_sequences % 100 == 0:
                    print(f"  Processed {processed_sequences} sequences, current L_norm = {log_likelihood_norm:.3f}")
        
        except Exception as e:
            print(f"Warning: Failed to process sequence: {e}")
            continue
    
    print(f"Successfully processed {len(log_likelihoods_norm)} validation sequences")
    
    if len(log_likelihoods_norm) < 10:
        raise ValueError(f"Insufficient validation data: only {len(log_likelihoods_norm)} sequences processed")
    
    # Compute universal reference statistics
    mu_ref = float(np.mean(log_likelihoods_norm))
    sigma_ref = float(np.std(log_likelihoods_norm))
    
    # Prevent division by zero
    sigma_ref = max(sigma_ref, 0.1)
    
    print(f"\nCalibrated universal reference statistics:")
    print(f"  μ_ref (mean): {mu_ref:.6f}")
    print(f"  σ_ref (std):  {sigma_ref:.6f}")
    print(f"  Min L_norm:   {np.min(log_likelihoods_norm):.6f}")
    print(f"  Max L_norm:   {np.max(log_likelihoods_norm):.6f}")
    print()
    
    # Create and save calibrated baseline
    baseline = UniversalReferenceBaseline()
    
    # Set calibrated parameters
    baseline._mu_ref = mu_ref
    baseline._sigma_ref = sigma_ref
    baseline._calibrated = True
    
    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.save(output_path)
    
    print(f"✓ Saved calibrated reference baseline to {output_path}")
    
    # Verification - test score computation
    print("\nVerification - testing score computation:")
    test_l_norms = [mu_ref, mu_ref - sigma_ref, mu_ref + sigma_ref]
    for l_norm in test_l_norms:
        # Research formula: Score = 80 + 10 × (L_norm - μ) / σ
        score_raw = 80 + 10 * (l_norm - mu_ref) / sigma_ref
        score_clamped = max(0.0, min(100.0, score_raw))
        print(f"  L_norm = {l_norm:.3f} → Score = {score_clamped:.1f}")
    
    print(f"\n=== Calibration Complete ===")
    print(f"Processed validation sequences: {len(log_likelihoods_norm)}")
    print(f"Universal μ_ref: {mu_ref:.6f}")
    print(f"Universal σ_ref: {sigma_ref:.6f}")
    print(f"Baseline saved to: {output_path}")
    
    return {
        'mu_ref': mu_ref,
        'sigma_ref': sigma_ref,
        'n_sequences': len(log_likelihoods_norm),
        'output_path': str(output_path),
        'l_norm_stats': {
            'min': float(np.min(log_likelihoods_norm)),
            'max': float(np.max(log_likelihoods_norm)),
            'percentiles': {
                '25': float(np.percentile(log_likelihoods_norm, 25)),
                '50': float(np.percentile(log_likelihoods_norm, 50)),
                '75': float(np.percentile(log_likelihoods_norm, 75))
            }
        }
    }


def test_calibrated_baseline(baseline_path: str) -> Dict[str, any]:
    """Test loading and using calibrated baseline.
    
    Args:
        baseline_path: Path to saved baseline file.
        
    Returns:
        Dict with test results.
    """
    print(f"Testing calibrated baseline from {baseline_path}...")
    
    # Load baseline
    baseline = UniversalReferenceBaseline()
    baseline.load(baseline_path)
    
    if not baseline.is_calibrated:
        raise RuntimeError("Baseline is not calibrated")
    
    stats = baseline.get_universal_statistics()
    print(f"Loaded baseline: μ_ref = {stats['mu']:.6f}, σ_ref = {stats['sigma']:.6f}")
    
    # Test VisualScorer integration
    scorer = VisualScorer(reference=baseline)
    
    # Mock HMM and features for testing
    feature_dimension = feature_dim(len(ALL_LIP_INDICES))
    hmm = GaussianHMM(num_states=1, feature_dim=feature_dimension)
    mock_features = np.random.randn(10, feature_dimension) * 0.1  # Small random features
    
    result = scorer.score(hmm, mock_features, "test")
    
    print(f"Test score result: {result['score']:.1f}")
    print(f"L_norm: {result.get('log_likelihood_norm', 'N/A')}")
    
    return {
        'baseline_loaded': True,
        'mu_ref': stats['mu'],
        'sigma_ref': stats['sigma'],
        'test_score': result['score']
    }


def main():
    """Command-line interface for reference baseline calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate reference baseline from GRID Corpus validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Calibrate using trained emissions (full dataset)
    python -m pronun.training.calibrate_reference /data/grid_corpus --emissions models/hmm_emissions.npz --output models/reference_baseline.npz
    
    # Calibrate with limited data (fast development)
    python -m pronun.training.calibrate_reference /data/grid_corpus --emissions models/hmm_emissions.npz --max-videos 50 --max-speakers 2
    
    # Test saved baseline
    python -m pronun.training.calibrate_reference --test-only --baseline models/reference_baseline.npz
        """
    )
    
    parser.add_argument('corpus_path', nargs='?', help='Path to GRID corpus root directory')
    parser.add_argument('--emissions', '-e', help='Path to trained HMM emissions file')
    parser.add_argument('--output', '-o', default='models/reference_baseline.npz',
                       help='Output path for calibrated baseline (default: models/reference_baseline.npz)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction for training, rest for calibration (default: 0.8)')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Maximum number of videos to use (default: all)')
    parser.add_argument('--max-speakers', type=int, default=None,
                       help='Maximum number of speakers to use (default: all)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing baseline (use with --baseline)')
    parser.add_argument('--baseline', help='Path to baseline file for testing')
    
    args = parser.parse_args()
    
    try:
        if args.test_only:
            # Test mode
            baseline_path = args.baseline or args.output
            if not Path(baseline_path).exists():
                print(f"Error: Baseline file not found at {baseline_path}")
                sys.exit(1)
            
            results = test_calibrated_baseline(baseline_path)
            print(f"\n✓ Baseline test completed successfully!")
            
        else:
            # Calibration mode
            if not args.corpus_path:
                print("Error: corpus_path required for calibration mode")
                sys.exit(1)
            
            if not args.emissions:
                print("Error: --emissions required for calibration mode")
                sys.exit(1)
            
            # Validate paths
            corpus_path = Path(args.corpus_path)
            if not corpus_path.exists():
                print(f"Error: GRID corpus not found at {corpus_path}")
                sys.exit(1)
            
            emissions_path = Path(args.emissions)
            if not emissions_path.exists():
                print(f"Error: Emissions file not found at {emissions_path}")
                sys.exit(1)
            
            if not (0.1 <= args.train_split <= 0.9):
                print(f"Error: train-split must be between 0.1 and 0.9")
                sys.exit(1)
            
            # Calibrate baseline
            results = calibrate_reference_baseline(
                corpus_path=str(corpus_path),
                emissions_path=str(emissions_path),
                output_path=args.output,
                train_split=args.train_split,
                max_videos=args.max_videos,
                max_speakers=args.max_speakers
            )
            
            print(f"\n✓ Reference baseline calibration completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()