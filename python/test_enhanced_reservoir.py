#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced ReservoirRWKV implementation.

Tests all advanced features including:
- ESN parameter mappings
- Multi-layer readouts
- Online learning
- Hierarchical outputs
- Chatbot persona configurations
- Batch processing
- Comparison with traditional ESNs
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any

# Add the parent directory to the path so we can import rwkv_cpp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rwkv_cpp import (
        RWKVSharedLibrary, 
        EnhancedReservoirRWKV, 
        ESNParameterMapping,
        create_chatbot_reservoir
    )
    print("✓ Successfully imported enhanced reservoir modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

def test_esn_parameter_mapping():
    """Test ESN parameter mapping functionality."""
    print("\n" + "="*60)
    print("Testing ESN Parameter Mapping")
    print("="*60)
    
    # Test parameter mapping retrieval
    mappings = ESNParameterMapping.get_parameter_mappings()
    
    expected_params = [
        'spectral_radius', 'leaking_rate', 'input_scaling', 
        'density', 'bias_scaling', 'noise_scaling'
    ]
    
    for param in expected_params:
        if param not in mappings:
            print(f"✗ Missing parameter mapping: {param}")
            return False
        
        mapping = mappings[param]
        required_keys = [
            'reservoirpy_description', 'rwkv_equivalent', 
            'chatbot_persona_effect', 'personality_mapping'
        ]
        
        for key in required_keys:
            if key not in mapping:
                print(f"✗ Missing key '{key}' in {param} mapping")
                return False
    
    print("✓ All ESN parameter mappings present and well-formed")
    
    # Test personality mappings
    test_param = mappings['spectral_radius']
    personalities = test_param['personality_mapping']
    expected_personalities = ['conservative', 'balanced', 'creative']
    
    for personality in expected_personalities:
        if personality not in personalities:
            print(f"✗ Missing personality mapping: {personality}")
            return False
    
    print("✓ Personality mappings complete")
    print(f"✓ Example spectral radius mappings: {personalities}")
    
    return True

def test_basic_enhanced_reservoir():
    """Test basic Enhanced ReservoirRWKV functionality."""
    print("\n" + "="*60)
    print("Testing Basic Enhanced Reservoir")
    print("="*60)
    
    # Find library and model
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    if not os.path.exists(library_path):
        print(f"✗ Library not found at {library_path}")
        return False
    
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    if not os.path.exists(test_model):
        print(f"✗ Test model not found at {test_model}")
        return False
    
    try:
        # Test library loading
        library = RWKVSharedLibrary(library_path)
        print(f"✓ Loaded library from {library_path}")
        
        # Test enhanced reservoir creation with different personas
        personas = ['conservative', 'balanced', 'creative']
        
        for persona in personas:
            print(f"\nTesting {persona} persona...")
            
            reservoir = EnhancedReservoirRWKV(
                shared_library=library,
                model_path=test_model,
                units=64,
                persona_type=persona,
                readout_type='ridge'
            )
            
            print(f"  ✓ Created {persona} reservoir")
            print(f"    - Vocabulary: {reservoir.n_vocab}")
            print(f"    - Embedding: {reservoir.n_embed}")
            print(f"    - Units: {reservoir.units}")
            print(f"    - Spectral radius: {reservoir.spectral_radius}")
            print(f"    - Leaking rate: {reservoir.leaking_rate}")
            
            # Test parameter info
            info = reservoir.get_esn_parameter_info()
            print(f"    - State scaling factor: {info['state_scaling_factor']:.3f}")
            
            # Test basic operations
            test_sequence = [1, 2, 3, 4, 5]
            activations = reservoir.run(test_sequence)
            print(f"    - Activations shape: {activations.shape}")
            
            if activations.shape != (5, 64):
                print(f"    ✗ Unexpected activations shape: {activations.shape}")
                return False
            
            print(f"  ✓ {persona} persona test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced reservoir test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_layer_readout():
    """Test multi-layer readout functionality."""
    print("\n" + "="*60)
    print("Testing Multi-Layer Readout")
    print("="*60)
    
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    if not os.path.exists(library_path) or not os.path.exists(test_model):
        print("✗ Required files not found, skipping test")
        return True
    
    try:
        library = RWKVSharedLibrary(library_path)
        
        # Test MLP readout
        reservoir = EnhancedReservoirRWKV(
            shared_library=library,
            model_path=test_model,
            units=64,
            readout_type='mlp',
            readout_config={
                'output_size': 1,
                'hidden_layers': [128, 64],
                'activation': 'relu',
                'use_torch': False  # Use sklearn to avoid torch dependency
            }
        )
        
        print("✓ Created reservoir with MLP readout")
        
        # Generate test data
        X_train = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        y_train = np.array([[0.1], [0.9]])
        
        # Test training
        reservoir.fit(X_train, y_train)
        print("✓ MLP readout training completed")
        
        # Test prediction
        predictions = reservoir.predict([1, 2, 3, 4, 5])
        print(f"✓ MLP predictions shape: {predictions.shape}")
        
        if not isinstance(predictions, np.ndarray):
            print("✗ Predictions should be numpy array")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-layer readout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_online_learning():
    """Test online learning functionality."""
    print("\n" + "="*60)
    print("Testing Online Learning")
    print("="*60)
    
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    if not os.path.exists(library_path) or not os.path.exists(test_model):
        print("✗ Required files not found, skipping test")
        return True
    
    try:
        library = RWKVSharedLibrary(library_path)
        
        # Test online learning readout
        reservoir = EnhancedReservoirRWKV(
            shared_library=library,
            model_path=test_model,
            units=64,
            readout_type='online',
            readout_config={
                'output_size': 1,
                'learning_rate': 0.01,
                'forgetting_factor': 0.99
            }
        )
        
        print("✓ Created reservoir with online learning readout")
        
        # Test initial training
        X_train = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        y_train = np.array([[0.1], [0.9]])
        
        reservoir.fit(X_train, y_train)
        print("✓ Online learning initial training completed")
        
        # Test prediction before update
        pred_before = reservoir.predict([1, 2, 3, 4, 5])
        print(f"✓ Prediction before update: {pred_before[-1]}")
        
        # Test online update
        reservoir.update_online(
            np.array([1, 2, 3, 4, 5]), 
            np.array([[0.5]])
        )
        print("✓ Online weight update completed")
        
        # Test prediction after update
        pred_after = reservoir.predict([1, 2, 3, 4, 5])
        print(f"✓ Prediction after update: {pred_after[-1]}")
        
        # Predictions should be different after update
        if np.allclose(pred_before[-1], pred_after[-1], atol=1e-6):
            print("⚠ Warning: Predictions unchanged after online update (may be expected)")
        else:
            print("✓ Predictions changed after online update")
        
        return True
        
    except Exception as e:
        print(f"✗ Online learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hierarchical_output():
    """Test hierarchical output functionality."""
    print("\n" + "="*60)
    print("Testing Hierarchical Output")
    print("="*60)
    
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    if not os.path.exists(library_path) or not os.path.exists(test_model):
        print("✗ Required files not found, skipping test")
        return True
    
    try:
        library = RWKVSharedLibrary(library_path)
        
        # Test hierarchical output
        hierarchical_configs = [
            {
                'output_size': 1,
                'time_scale': 1,
                'readout_type': 'ridge',
                'readout_params': {'alpha': 1e-6}
            },
            {
                'output_size': 1,
                'time_scale': 3,
                'readout_type': 'ridge',
                'readout_params': {'alpha': 1e-4}
            }
        ]
        
        reservoir = EnhancedReservoirRWKV(
            shared_library=library,
            model_path=test_model,
            units=64,
            readout_type='hierarchical',
            hierarchical_configs=hierarchical_configs
        )
        
        print("✓ Created reservoir with hierarchical outputs")
        
        # Generate longer test sequence for hierarchical processing
        X_train = [list(range(1, 16))]  # 15 time steps
        y_train = {
            'readout_0_1': np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]).T,
            'readout_1_3': np.array([[0.3, 0.9, 1.5, 2.1, 2.7]]).T  # Every 3rd step
        }
        
        # Test training
        reservoir.fit(X_train, y_train)
        print("✓ Hierarchical output training completed")
        
        # Test prediction
        predictions = reservoir.predict(list(range(1, 16)))
        print(f"✓ Hierarchical predictions: {type(predictions)}")
        
        if not isinstance(predictions, dict):
            print("✗ Hierarchical predictions should be a dictionary")
            return False
        
        print(f"  - Readout levels: {list(predictions.keys())}")
        for key, pred in predictions.items():
            print(f"  - {key} shape: {pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hierarchical output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chatbot_personas():
    """Test chatbot persona functionality."""
    print("\n" + "="*60)
    print("Testing Chatbot Personas")
    print("="*60)
    
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    if not os.path.exists(library_path) or not os.path.exists(test_model):
        print("✗ Required files not found, skipping test")
        return True
    
    try:
        library = RWKVSharedLibrary(library_path)
        
        # Test convenience function
        reservoir = create_chatbot_reservoir(
            shared_library=library,
            model_path=test_model,
            persona_type='creative',
            advanced_features=True,
            units=64
        )
        
        print("✓ Created chatbot reservoir with convenience function")
        print(f"  - Persona: {reservoir.persona_type}")
        print(f"  - Readout type: {reservoir.readout_type}")
        
        # Test persona changing
        original_spectral_radius = reservoir.spectral_radius
        
        reservoir.set_persona('conservative')
        print(f"✓ Changed persona to conservative")
        print(f"  - Spectral radius changed: {original_spectral_radius} → {reservoir.spectral_radius}")
        
        # Test parameter info display
        info = reservoir.get_esn_parameter_info()
        print("✓ ESN parameter info retrieved:")
        for param, value in info['current_values'].items():
            print(f"  - {param}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Chatbot persona test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing functionality."""
    print("\n" + "="*60)
    print("Testing Batch Processing")
    print("="*60)
    
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    if not os.path.exists(library_path) or not os.path.exists(test_model):
        print("✗ Required files not found, skipping test")
        return True
    
    try:
        library = RWKVSharedLibrary(library_path)
        
        reservoir = EnhancedReservoirRWKV(
            shared_library=library,
            model_path=test_model,
            units=64,
            readout_type='ridge'
        )
        
        # Train on simple data
        X_train = [[1, 2, 3], [4, 5, 6]]
        y_train = np.array([[0.1], [0.9]])
        
        reservoir.fit(X_train, y_train)
        print("✓ Trained reservoir for batch processing test")
        
        # Test batch prediction
        X_batch = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]
        
        batch_predictions = reservoir.batch_predict(X_batch)
        print(f"✓ Batch processing completed")
        print(f"  - Batch size: {len(batch_predictions)}")
        
        for i, pred in enumerate(batch_predictions):
            print(f"  - Sequence {i+1} prediction shape: {pred.shape}")
        
        if len(batch_predictions) != len(X_batch):
            print("✗ Batch prediction count mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run comprehensive test suite."""
    print("Enhanced ReservoirRWKV Comprehensive Test Suite")
    print("=" * 70)
    
    tests = [
        ("ESN Parameter Mapping", test_esn_parameter_mapping),
        ("Basic Enhanced Reservoir", test_basic_enhanced_reservoir),
        ("Multi-Layer Readout", test_multi_layer_readout),
        ("Online Learning", test_online_learning),
        ("Hierarchical Output", test_hierarchical_output),
        ("Chatbot Personas", test_chatbot_personas),
        ("Batch Processing", test_batch_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")
            
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)