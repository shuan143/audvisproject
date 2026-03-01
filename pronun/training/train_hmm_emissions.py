#!/usr/bin/env python3
"""Train HMM emission parameters from GRID Corpus data.

This script implements Task #2 from CLAUDE.md:
- Extract 254-dim lip features from GRID corpus videos
- Train Gaussian emission parameters for each viseme (0-12) 
- Save trained parameters to disk for Session loading

Usage:
    python -m pronun.training.train_hmm_emissions /path/to/grid/corpus --output models/hmm_emissions.npz
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Dict

from pronun.data.grid_corpus import GridCorpusDataset, GridCorpusFeatureExtractor
from pronun.visual.scoring.hmm import GaussianHMM
from pronun.visual.features.feature_builder import feature_dim
from pronun.config import ALL_LIP_INDICES


def train_hmm_emissions(corpus_path: str, output_path: str, train_split: float = 0.8, 
                       max_videos: int = None, max_speakers: int = None) -> Dict[str, any]:
    """Train HMM emission parameters from GRID corpus.
    
    Args:
        corpus_path: Path to GRID corpus root directory.
        output_path: Path to save trained emissions (.npz file).
        train_split: Fraction of data for training.
        max_videos: Maximum number of training videos to use (None = all).
        max_speakers: Maximum number of speakers to use (None = all).
        
    Returns:
        Dict with training statistics.
    """
    print("=== HMM Emission Training ===")
    print(f"Corpus path: {corpus_path}")
    print(f"Output path: {output_path}")
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
    print(f"Found {corpus_stats['total_speakers']} speakers")
    print(f"Total videos: {corpus_stats['total_videos']}")
    print(f"Training videos: {corpus_stats['train_videos']}")
    print(f"Validation videos: {corpus_stats['val_videos']}")
    print()
    
    # Get corpus statistics
    stats = dataset.get_corpus_statistics()
    print("Corpus statistics:")
    print(f"  Total viseme tokens: {stats['total_viseme_tokens']}")
    print(f"  Unique visemes: {stats['unique_visemes']}")
    print(f"  Avg transcript length: {stats['avg_transcript_length']:.1f} words")
    print("  Most common visemes:", stats['most_common_visemes'][:5])
    print()
    
    # Extract training features
    feature_extractor = GridCorpusFeatureExtractor(dataset)
    print("Extracting training features from videos...")
    print("This may take several minutes depending on corpus size...")
    viseme_features = feature_extractor.extract_training_features()
    
    print(f"\nExtracted features for {len(viseme_features)} visemes:")
    for viseme_id in sorted(viseme_features.keys()):
        n_samples = len(viseme_features[viseme_id])
        print(f"  Viseme {viseme_id}: {n_samples} feature vectors")
    print()
    
    # Determine feature dimension
    feature_dimension = feature_dim(len(ALL_LIP_INDICES))
    print(f"Feature dimension: {feature_dimension}")
    
    # Train emission parameters for each viseme
    print("Training Gaussian emission parameters...")
    trained_emissions = {}
    
    for viseme_id in sorted(viseme_features.keys()):
        observations = viseme_features[viseme_id]
        print(f"Training viseme {viseme_id} with {len(observations)} samples...")
        
        # Create single-state HMM for this viseme
        hmm = GaussianHMM(num_states=1, feature_dim=feature_dimension)
        hmm.train_emissions(0, observations)
        
        # Store parameters
        trained_emissions[viseme_id] = {
            'mean': hmm.means[0].copy(),
            'variance': hmm.variances[0].copy(),
            'n_samples': len(observations)
        }
    
    print(f"Trained emission parameters for {len(trained_emissions)} visemes")
    
    # Save trained parameters
    print(f"Saving trained emissions to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'feature_dimension': feature_dimension,
        'num_visemes': len(trained_emissions),
        'training_stats': {
            'total_videos': corpus_stats['train_videos'],
            'total_speakers': corpus_stats['total_speakers'],
            'viseme_distribution': {str(k): v for k, v in stats['viseme_distribution'].items()}
        }
    }
    
    # Add emission parameters
    for viseme_id, params in trained_emissions.items():
        save_data[f'viseme_{viseme_id}_mean'] = params['mean']
        save_data[f'viseme_{viseme_id}_variance'] = params['variance']
        save_data[f'viseme_{viseme_id}_n_samples'] = params['n_samples']
    
    np.savez_compressed(output_path, **save_data)
    print(f"✓ Saved trained HMM emissions to {output_path}")
    
    # Training summary
    total_samples = sum(params['n_samples'] for params in trained_emissions.values())
    print(f"\n=== Training Complete ===")
    print(f"Trained visemes: {list(sorted(trained_emissions.keys()))}")
    print(f"Total training samples: {total_samples}")
    print(f"Feature dimension: {feature_dimension}")
    print(f"Model saved to: {output_path}")
    
    return {
        'trained_visemes': list(sorted(trained_emissions.keys())),
        'total_samples': total_samples,
        'feature_dimension': feature_dimension,
        'output_path': str(output_path)
    }


def load_trained_emissions(model_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Load trained HMM emission parameters from disk.
    
    Args:
        model_path: Path to saved emissions (.npz file).
        
    Returns:
        Dict mapping viseme_id -> {'mean': array, 'variance': array, 'n_samples': int}
    """
    data = np.load(model_path)
    
    emissions = {}
    viseme_ids = []
    
    # Extract viseme IDs from keys
    for key in data.keys():
        if key.startswith('viseme_') and key.endswith('_mean'):
            viseme_id = int(key.split('_')[1])
            viseme_ids.append(viseme_id)
    
    # Load parameters for each viseme
    for viseme_id in sorted(set(viseme_ids)):
        mean_key = f'viseme_{viseme_id}_mean'
        var_key = f'viseme_{viseme_id}_variance'
        n_samples_key = f'viseme_{viseme_id}_n_samples'
        
        if all(key in data for key in [mean_key, var_key, n_samples_key]):
            emissions[viseme_id] = {
                'mean': data[mean_key],
                'variance': data[var_key],
                'n_samples': int(data[n_samples_key])
            }
    
    print(f"Loaded emission parameters for visemes: {list(sorted(emissions.keys()))}")
    return emissions


def main():
    """Command-line interface for HMM emission training."""
    parser = argparse.ArgumentParser(
        description="Train HMM emission parameters from GRID Corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train from GRID corpus (full dataset)
    python -m pronun.training.train_hmm_emissions /data/grid_corpus --output models/hmm_emissions.npz
    
    # Train with limited data (fast development)
    python -m pronun.training.train_hmm_emissions /data/grid_corpus --max-videos 100 --max-speakers 3
    
    # Custom train/validation split
    python -m pronun.training.train_hmm_emissions /data/grid_corpus --train-split 0.85
        """
    )
    
    parser.add_argument('corpus_path', help='Path to GRID corpus root directory')
    parser.add_argument('--output', '-o', default='models/hmm_emissions.npz',
                       help='Output path for trained emissions (default: models/hmm_emissions.npz)')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training (default: 0.8)')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Maximum number of videos to use (default: all)')
    parser.add_argument('--max-speakers', type=int, default=None,
                       help='Maximum number of speakers to use (default: all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"Error: GRID corpus not found at {corpus_path}")
        sys.exit(1)
    
    if not (0.1 <= args.train_split <= 0.9):
        print(f"Error: train-split must be between 0.1 and 0.9")
        sys.exit(1)
    
    try:
        # Train emissions
        results = train_hmm_emissions(
            corpus_path=str(corpus_path),
            output_path=args.output,
            train_split=args.train_split,
            max_videos=args.max_videos,
            max_speakers=args.max_speakers
        )
        
        print(f"\n✓ HMM emission training completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()