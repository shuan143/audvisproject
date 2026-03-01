#!/usr/bin/env python3
"""Example usage of trained HMM emissions and calibrated reference baseline.

This demonstrates how to use Session with trained visual scoring components.
"""

from pathlib import Path
from pronun.workflow.session import Session

def main():
    """Example of using Session with trained models."""
    
    # Paths to trained models (adjust as needed)
    emissions_path = "models/hmm_emissions.npz"
    baseline_path = "models/reference_baseline.npz"
    
    # Check if models exist
    models_available = True
    if not Path(emissions_path).exists():
        print(f"Warning: Trained emissions not found at {emissions_path}")
        models_available = False
    
    if not Path(baseline_path).exists():
        print(f"Warning: Calibrated baseline not found at {baseline_path}")
        models_available = False
    
    if models_available:
        print("=== Using Trained Visual Scoring Models ===")
        # Create session with trained models
        session = Session(
            use_camera=True,
            mode="B",  # Mode B (Lee viseme mapping)
            hmm_emissions_path=emissions_path,
            reference_baseline_path=baseline_path
        )
    else:
        print("=== Using Untrained Models (visual score will be 0.0) ===")
        # Fallback to default session
        session = Session(use_camera=True, mode="B")
    
    try:
        session.setup()
        
        # Test with a simple word
        test_word = "hello"
        print(f"\nTesting pronunciation of: '{test_word}'")
        print("Speak when recording starts...")
        
        result = session.practice_word(test_word)
        
        print("\n=== Results ===")
        print(f"Audio Score: {result['audio_score']:.1f}")
        print(f"Visual Score: {result.get('visual_score', 'N/A')}")
        print(f"Combined Score: {result['combined_score']:.1f}")
        
        if result.get('visual_details_b'):
            details = result['visual_details_b']
            print(f"L_norm: {details.get('log_likelihood_norm', 'N/A'):.6f}")
            print(f"μ_ref: {details.get('mu_ref', 'N/A'):.6f}")
            print(f"σ_ref: {details.get('sigma_ref', 'N/A'):.6f}")
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        session.teardown()


if __name__ == '__main__':
    main()