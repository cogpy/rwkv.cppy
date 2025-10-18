#!/usr/bin/env python3
"""
Example of using ReservoirRWKV for time series prediction.
This demonstrates the ReservoirPy-compatible API.
"""

import sys
import os
import numpy as np
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the current directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from rwkv_cpp import RWKVSharedLibrary, ReservoirRWKV


def generate_sine_sequences(
    n_sequences: int = 50,
    seq_len: int = 100,
    vocab_size: int = 256,
    noise_level: float = 0.1
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generate sine wave sequences as token sequences for time series prediction.
    
    Returns
    -------
    sequences : List[List[int]]
        Token sequences representing discretized sine waves
    targets : np.ndarray
        Next values in the sine wave sequence
    """
    sequences = []
    targets = []
    
    for i in range(n_sequences):
        # Generate sine wave with random phase and frequency
        phase = np.random.random() * 2 * np.pi
        freq = 0.1 + np.random.random() * 0.2  # Frequency between 0.1 and 0.3
        
        # Generate sine wave values
        t = np.arange(seq_len + 1)
        sine_values = np.sin(2 * np.pi * freq * t + phase)
        
        # Add noise
        sine_values += np.random.normal(0, noise_level, len(sine_values))
        
        # Discretize to tokens (map [-1, 1] to [0, vocab_size-1])
        tokens = ((sine_values + 1) / 2 * (vocab_size - 1)).astype(int)
        tokens = np.clip(tokens, 0, vocab_size - 1)
        
        # Input sequence and target (next value)
        input_seq = tokens[:-1].tolist()
        target = sine_values[-1]  # Use continuous target for regression
        
        sequences.append(input_seq)
        targets.append([target])
    
    return sequences, np.array(targets, dtype=np.float32)


def generate_memory_task(
    n_sequences: int = 100,
    seq_len: int = 50,
    vocab_size: int = 10,
    memory_len: int = 5
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Generate a simple memory task: predict the sum of the first memory_len tokens.
    
    Returns
    -------
    sequences : List[List[int]]
        Input token sequences
    targets : np.ndarray
        Sum of first memory_len tokens (normalized)
    """
    sequences = []
    targets = []
    
    for _ in range(n_sequences):
        # Generate random sequence
        seq = np.random.randint(0, vocab_size, size=seq_len).tolist()
        
        # Target is sum of first memory_len tokens (normalized)
        target_sum = sum(seq[:memory_len]) / (vocab_size * memory_len)
        
        sequences.append(seq)
        targets.append([target_sum])
    
    return sequences, np.array(targets, dtype=np.float32)


def run_sine_prediction_example():
    """Example: Sine wave prediction using ReservoirRWKV."""
    print("Sine Wave Prediction Example")
    print("-" * 40)
    
    # Find library and model
    library_path = None
    possible_paths = ['../librwkv.so', '../librwkv.dylib', '../librwkv.dll']
    
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            library_path = full_path
            break
    
    if library_path is None:
        print("Error: Could not find librwkv shared library")
        return False
    
    # Find test model
    test_dir = os.path.join(os.path.dirname(__file__), '../tests')
    model_path = None
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('.bin') and 'tiny-rwkv' in file and 'FP32' in file:
                model_path = os.path.join(test_dir, file)
                break
    
    if model_path is None:
        print("Error: Could not find test model")
        return False
    
    print(f"Using model: {model_path}")
    
    try:
        # Initialize library and reservoir
        library = RWKVSharedLibrary(library_path)
        reservoir = ReservoirRWKV(
            shared_library=library,
            model_path=model_path,
            units=128,
            alpha=1e-4,
            thread_count=1
        )
        
        print(f"Reservoir initialized with {reservoir.units} units")
        
        # Generate training data
        print("Generating sine wave data...")
        X_train, y_train = generate_sine_sequences(n_sequences=30, seq_len=50)
        X_test, y_test = generate_sine_sequences(n_sequences=10, seq_len=50)
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        
        # Train the reservoir
        print("Training reservoir...")
        reservoir.fit(X_train, y_train, warmup=10)
        print("Training completed")
        
        # Evaluate performance
        train_score = reservoir.score(X_train, y_train, warmup=10)
        test_score = reservoir.score(X_test, y_test, warmup=10)
        
        print(f"Training R^2: {train_score:.4f}")
        print(f"Test R^2: {test_score:.4f}")
        
        # Make predictions on test set
        print("\nMaking predictions...")
        test_predictions = []
        for seq in X_test[:5]:  # First 5 test sequences
            pred = reservoir.predict(seq)
            test_predictions.append(pred[-1])  # Last prediction
        
        print("Sample predictions vs targets:")
        for i, (pred, true) in enumerate(zip(test_predictions, y_test[:5].flatten())):
            print(f"  Sequence {i+1}: pred={pred:.4f}, true={true:.4f}, error={abs(pred-true):.4f}")
        
        reservoir.rwkv_model.free()
        return True
        
    except Exception as e:
        print(f"Error in sine prediction example: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_memory_task_example():
    """Example: Memory task using ReservoirRWKV."""
    print("\nMemory Task Example")
    print("-" * 40)
    
    # Find library and model (same as above)
    library_path = None
    possible_paths = ['../librwkv.so', '../librwkv.dylib', '../librwkv.dll']
    
    for path in possible_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            library_path = full_path
            break
    
    if library_path is None:
        print("Error: Could not find librwkv shared library")
        return False
    
    test_dir = os.path.join(os.path.dirname(__file__), '../tests')
    model_path = None
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('.bin') and 'tiny-rwkv' in file and 'FP32' in file:
                model_path = os.path.join(test_dir, file)
                break
    
    if model_path is None:
        print("Error: Could not find test model")
        return False
    
    try:
        # Initialize reservoir
        library = RWKVSharedLibrary(library_path)
        reservoir = ReservoirRWKV(
            shared_library=library,
            model_path=model_path,
            units=64,
            alpha=1e-3,
            thread_count=1
        )
        
        print(f"Reservoir initialized for memory task")
        
        # Generate memory task data
        print("Generating memory task data...")
        X_train, y_train = generate_memory_task(n_sequences=50, seq_len=30, memory_len=3)
        X_test, y_test = generate_memory_task(n_sequences=20, seq_len=30, memory_len=3)
        
        print(f"Task: Predict sum of first 3 tokens")
        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        
        # Train the reservoir
        print("Training reservoir...")
        reservoir.fit(X_train, y_train, warmup=5)
        
        # Evaluate performance  
        train_score = reservoir.score(X_train, y_train, warmup=5)
        test_score = reservoir.score(X_test, y_test, warmup=5)
        
        print(f"Training R^2: {train_score:.4f}")
        print(f"Test R^2: {test_score:.4f}")
        
        # Show some examples
        print("\nSample predictions vs targets:")
        for i in range(5):
            seq = X_test[i]
            true_sum = sum(seq[:3]) / (10 * 3)  # Normalized sum
            pred = reservoir.predict(seq)[-1]  # Last prediction
            
            print(f"  Sequence {i+1}: {seq[:3]} -> pred={pred:.4f}, true={true_sum:.4f}, error={abs(pred-true_sum):.4f}")
        
        reservoir.rwkv_model.free()
        return True
        
    except Exception as e:
        print(f"Error in memory task example: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run reservoir computing examples."""
    print("ReservoirRWKV Examples")
    print("=" * 50)
    
    success = True
    
    # Run sine prediction example
    if not run_sine_prediction_example():
        success = False
    
    # Run memory task example  
    if not run_memory_task_example():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All examples completed successfully!")
        return 0
    else:
        print("✗ Some examples failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())