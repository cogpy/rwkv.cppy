#!/usr/bin/env python3
"""
Test script for ReservoirRWKV implementation.
This demonstrates the ReservoirPy-compatible API using RWKV as a reservoir.
"""

import sys
import os
import numpy as np
from typing import List

# Add the current directory to path so we can import rwkv_cpp
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from rwkv_cpp import RWKVSharedLibrary, ReservoirRWKV
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the rwkv.cpp library is built and available.")
    sys.exit(1)


def generate_test_sequences(n_sequences: int = 100, seq_len: int = 50, vocab_size: int = 1000) -> List[List[int]]:
    """Generate random test sequences."""
    sequences = []
    for _ in range(n_sequences):
        # Generate random token sequence
        seq = np.random.randint(0, vocab_size, size=seq_len).tolist()
        sequences.append(seq)
    return sequences


def generate_test_targets_classification(sequences: List[List[int]]) -> np.ndarray:
    """Generate test targets for classification task based on sequence properties."""
    targets = []
    for seq in sequences:
        # Simple classification: even/odd based on sequence sum
        target = (sum(seq) % 2)
        targets.append([target])
    return np.array(targets, dtype=np.float32)


def generate_test_targets_regression(sequences: List[List[int]]) -> np.ndarray:
    """Generate test targets for regression task based on sequence properties."""
    targets = []
    for seq in sequences:
        # Simple regression: predict normalized sequence mean
        target = np.mean(seq) / 1000.0  # Normalize to [0, 1]
        targets.append([target])
    return np.array(targets, dtype=np.float32)


def test_reservoir_basic():
    """Test basic reservoir functionality."""
    print("Testing basic ReservoirRWKV functionality...")
    
    # Find library
    library_path = None
    possible_paths = [
        '../librwkv.so',
        '../lib/librwkv.so',
        '../build/librwkv.so',
        '../librwkv.dylib',
        '../librwkv.dll'
    ]
    
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            library_path = full_path
            break
    
    if library_path is None:
        print("Error: Could not find librwkv shared library")
        print("Please build the project first with: cmake . && cmake --build . --config Release")
        return False
    
    try:
        library = RWKVSharedLibrary(library_path)
        print(f"✓ Loaded library from {library_path}")
    except Exception as e:
        print(f"✗ Failed to load library: {e}")
        return False
    
    # Check for test model (we'll create a dummy path for testing the interface)
    test_model_path = "/nonexistent/path/test.bin"  # This will fail gracefully
    
    try:
        # This should fail but test the interface
        reservoir = ReservoirRWKV(
            shared_library=library,
            model_path=test_model_path,
            units=128,
            alpha=1e-3
        )
        print("✗ Expected model loading to fail with non-existent path")
        return False
    except (FileNotFoundError, ValueError) as e:
        print(f"✓ Correctly handled non-existent model path: {type(e).__name__}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("✓ Basic interface test passed")
    return True


def test_reservoir_with_model():
    """Test with actual model if available."""
    print("\nTesting with actual model (if available)...")
    
    # Look for test models
    test_models = []
    test_dir = os.path.join(os.path.dirname(__file__), '../tests')
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('.bin') and 'tiny-rwkv' in file and 'FP32' in file:
                test_models.append(os.path.join(test_dir, file))
    
    if not test_models:
        print("No test models found, skipping model tests")
        return True
    
    model_path = test_models[0]
    print(f"Using test model: {model_path}")
    
    # Find library
    library_path = None
    possible_paths = [
        '../librwkv.so',
        '../librwkv.dylib', 
        '../librwkv.dll'
    ]
    
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            library_path = full_path
            break
    
    if library_path is None:
        print("Library not found, skipping model tests")
        return True
    
    try:
        library = RWKVSharedLibrary(library_path)
        
        # Create reservoir
        reservoir = ReservoirRWKV(
            shared_library=library,
            model_path=model_path,
            units=64,  # Use smaller number for test
            alpha=1e-3,
            thread_count=1
        )
        
        print(f"✓ Successfully created reservoir")
        print(f"  - Vocabulary size: {reservoir.n_vocab}")
        print(f"  - Embedding size: {reservoir.n_embed}")
        print(f"  - Number of layers: {reservoir.n_layer}")
        print(f"  - Reservoir units: {reservoir.units}")
        
        # Test basic operations
        test_sequence = [1, 2, 3, 4, 5]  # Simple test sequence
        
        # Test run (get activations without training)
        activations = reservoir.run(test_sequence)
        print(f"✓ Got activations with shape: {activations.shape}")
        
        # Test training with dummy data - single target per sequence
        X_train = [test_sequence] * 10  # 10 copies of test sequence 
        y_train = np.random.random((10, 1)).astype(np.float32)
        
        reservoir.fit(X_train, y_train)
        print("✓ Training completed")
        
        # Test prediction
        predictions = reservoir.predict(test_sequence)
        print(f"✓ Got predictions with shape: {predictions.shape}")
        
        # Test scoring
        score = reservoir.score(X_train, y_train)
        print(f"✓ R^2 score: {score:.4f}")
        
        reservoir.rwkv_model.free()
        print("✓ Model tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("ReservoirRWKV Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test basic functionality
    if not test_reservoir_basic():
        success = False
    
    # Test with actual model
    if not test_reservoir_with_model():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())