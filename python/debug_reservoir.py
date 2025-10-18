#!/usr/bin/env python3
"""
Debug script for ReservoirRWKV implementation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from rwkv_cpp import RWKVSharedLibrary, ReservoirRWKV

def debug_shapes():
    """Debug the shapes in reservoir implementation."""
    print("Debugging ReservoirRWKV shapes...")
    
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
        library = RWKVSharedLibrary(library_path)
        reservoir = ReservoirRWKV(
            shared_library=library,
            model_path=model_path,
            units=32,  # Small for debugging
            alpha=1e-3,
            thread_count=1
        )
        
        print(f"Reservoir created: {reservoir.units} units")
        
        # Test with simple data
        test_sequence = [1, 2, 3]
        print(f"Test sequence: {test_sequence}")
        
        # Test run
        activations = reservoir.run(test_sequence)
        print(f"Activations shape: {activations.shape}")
        
        # Test training with single sequence
        X_single = test_sequence
        y_single = np.array([[0.5]])
        print(f"Single X shape: {len(X_single)} (length), y shape: {y_single.shape}")
        
        reservoir.fit(X_single, y_single)
        print("✓ Single sequence training works")
        
        # Test prediction on single sequence
        pred_single = reservoir.predict(test_sequence)
        print(f"Single prediction shape: {pred_single.shape}")
        
        # Test with multiple sequences
        X_multi = [test_sequence, [4, 5, 6], [7, 8, 9]]
        y_multi = np.array([[0.1], [0.5], [0.9]])
        print(f"Multi X: {len(X_multi)} sequences, y shape: {y_multi.shape}")
        
        reservoir.fit(X_multi, y_multi)
        print("✓ Multi sequence training works")
        
        # Test predictions on multiple sequences
        for i, seq in enumerate(X_multi):
            pred = reservoir.predict(seq)
            print(f"Sequence {i} prediction shape: {pred.shape}")
        
        # Test scoring
        print("Debugging score method...")
        try:
            score = reservoir.score(X_multi, y_multi)
            print(f"Score: {score:.4f}")
        except Exception as e:
            print(f"Score error: {e}")
            
            # Debug shapes manually
            all_predictions = []
            all_targets = []
            
            for i, (seq, target) in enumerate(zip(X_multi, y_multi)):
                pred = reservoir.predict(seq, reset_state=True)
                print(f"Seq {i}: pred shape {pred.shape}, target shape {target.shape}")
                all_predictions.append(pred)
                
                # Repeat target for all time steps
                seq_targets = np.repeat(target.reshape(1, -1), len(pred), axis=0)
                all_targets.append(seq_targets)
                print(f"  -> expanded target shape: {seq_targets.shape}")
                
            print(f"Individual prediction shapes: {[p.shape for p in all_predictions]}")
            print(f"Individual target shapes: {[t.shape for t in all_targets]}")
            
            y_pred_all = np.concatenate(all_predictions)
            y_true_all = np.vstack(all_targets)
            print(f"Final shapes: pred {y_pred_all.shape}, true {y_true_all.shape}")
            
            # Flatten if needed
            if y_pred_all.ndim > 1 and y_pred_all.shape[1] == 1:
                y_pred_flat = y_pred_all.flatten()
                print(f"Flattened pred shape: {y_pred_flat.shape}")
            else:
                y_pred_flat = y_pred_all
                
            if y_true_all.ndim > 1 and y_true_all.shape[1] == 1:
                y_true_flat = y_true_all.flatten()
                print(f"Flattened true shape: {y_true_flat.shape}")
            else:
                y_true_flat = y_true_all
            
            from sklearn.metrics import r2_score
            score = r2_score(y_true_flat, y_pred_flat)
            print(f"Manual score: {score:.4f}")
        
        reservoir.rwkv_model.free()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_shapes()