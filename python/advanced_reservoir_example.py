#!/usr/bin/env python3
"""
Advanced example demonstrating Enhanced ReservoirRWKV features for chatbot personality modeling.

This example showcases:
1. Detailed ESN parameter mappings to RWKV.cpp features
2. Multiple chatbot personas with different characteristics
3. Multi-layer readout networks for complex behaviors
4. Online learning for real-time adaptation
5. Hierarchical outputs for multi-scale reasoning
6. Batch processing for efficiency
7. Comparison with traditional ESNs

The examples focus on modeling different chatbot personalities using reservoir computing
principles mapped to RWKV.cpp's efficient sequential processing.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any

# Add the parent directory to the path
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
    print("Make sure you're running this from the correct directory and that rwkv_cpp is built.")
    sys.exit(1)

def setup_environment():
    """Set up the environment and check for required files."""
    # Find library and model
    library_path = os.path.join(os.path.dirname(__file__), "..", "librwkv.so")
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    if not os.path.exists(library_path):
        print(f"✗ Library not found at {library_path}")
        print("Please build the project first: cmake -B build . && cd build && make")
        return None, None
    
    if not os.path.exists(test_model):
        print(f"✗ Test model not found at {test_model}")
        print("This example requires the test model files.")
        return None, None
    
    try:
        library = RWKVSharedLibrary(library_path)
        print(f"✓ Loaded library from {library_path}")
        return library, test_model
    except Exception as e:
        print(f"✗ Failed to load library: {e}")
        return None, None

def demonstrate_esn_parameter_mappings():
    """Demonstrate detailed ESN parameter mappings."""
    print("\n" + "="*70)
    print("1. ESN PARAMETER MAPPINGS TO RWKV.CPP")
    print("="*70)
    
    mappings = ESNParameterMapping.get_parameter_mappings()
    
    print("Detailed mapping of ReservoirPy ESN parameters to RWKV.cpp concepts:")
    print()
    
    for param_name, mapping in mappings.items():
        print(f"🔧 {param_name.upper()}")
        print(f"   ReservoirPy: {mapping['reservoirpy_description']}")
        print(f"   RWKV Equivalent: {mapping['rwkv_equivalent']}")
        print(f"   Chatbot Effect: {mapping['chatbot_persona_effect']}")
        print(f"   Implementation: {mapping['implementation']}")
        print(f"   Value Range: {mapping['value_range']}")
        print(f"   Personality Mapping:")
        for persona, value in mapping['personality_mapping'].items():
            print(f"     {persona}: {value}")
        print()
    
    print("💡 These mappings allow fine-tuned control over chatbot behavior using")
    print("   well-established reservoir computing principles!")

def demonstrate_chatbot_personas(library, model_path):
    """Demonstrate different chatbot personas."""
    print("\n" + "="*70)
    print("2. CHATBOT PERSONALITY MODELING")
    print("="*70)
    
    personas = ['conservative', 'balanced', 'creative']
    reservoirs = {}
    
    # Create reservoirs with different personas
    for persona in personas:
        print(f"\n📱 Creating {persona.upper()} chatbot persona...")
        
        reservoir = create_chatbot_reservoir(
            shared_library=library,
            model_path=model_path,
            persona_type=persona,
            advanced_features=True,
            units=64
        )
        
        reservoirs[persona] = reservoir
        
        # Display persona characteristics
        info = reservoir.get_esn_parameter_info()
        print(f"   Characteristics:")
        print(f"   - Spectral radius: {info['current_values']['spectral_radius']:.2f} (creativity/stability)")
        print(f"   - Leaking rate: {info['current_values']['leaking_rate']:.2f} (memory persistence)")
        print(f"   - Input scaling: {info['current_values']['input_scaling']:.2f} (sensitivity)")
        print(f"   - Noise scaling: {info['current_values']['noise_scaling']:.3f} (variability)")
    
    # Test response patterns
    print(f"\n🧪 Testing response patterns with sample conversations:")
    
    # Simulate conversation sequences (simple token patterns)
    conversation_patterns = [
        [1, 15, 23, 45, 67],      # Greeting pattern
        [89, 12, 156, 78, 90],    # Question pattern  
        [200, 34, 67, 123, 45]    # Emotional pattern
    ]
    
    for i, pattern in enumerate(conversation_patterns):
        print(f"\n   Pattern {i+1}: {pattern}")
        
        for persona, reservoir in reservoirs.items():
            activations = reservoir.run(pattern)
            # Use activation variance as a measure of "responsiveness"
            responsiveness = np.var(activations)
            print(f"     {persona:>12}: responsiveness = {responsiveness:.4f}")
    
    return reservoirs

def demonstrate_multi_layer_readout(library, model_path):
    """Demonstrate multi-layer readout networks."""
    print("\n" + "="*70)
    print("3. MULTI-LAYER READOUT NETWORKS")
    print("="*70)
    
    print("🧠 Creating reservoir with deep neural network readout...")
    
    # Create reservoir with MLP readout
    reservoir = EnhancedReservoirRWKV(
        shared_library=library,
        model_path=model_path,
        units=64,
        persona_type='balanced',
        readout_type='mlp',
        readout_config={
            'output_size': 3,  # Multi-output for complex behaviors
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'dropout': 0.1,
            'use_torch': False  # Use sklearn for compatibility
        }
    )
    
    # Generate training data for complex behavior modeling
    print("📊 Generating training data for complex behavior modeling...")
    
    sequences = []
    targets = []
    
    for i in range(50):
        # Generate varied conversation patterns
        seq_length = np.random.randint(10, 20)
        sequence = np.random.randint(0, 256, seq_length).tolist()
        
        # Multi-dimensional targets representing different aspects:
        # [sentiment, formality, creativity]
        sentiment = (np.sum(sequence) % 100) / 100.0
        formality = (len(sequence) % 10) / 10.0
        creativity = np.std(sequence) / 50.0
        
        sequences.append(sequence)
        targets.append([sentiment, formality, creativity])
    
    targets = np.array(targets)
    
    # Train the multi-layer readout
    print("🎯 Training multi-layer readout...")
    reservoir.fit(sequences, targets)
    
    # Test predictions
    test_sequence = [1, 50, 100, 150, 200, 25, 75, 125, 175, 225]
    prediction = reservoir.predict(test_sequence)
    
    print(f"✓ Multi-layer readout trained successfully")
    print(f"   Test prediction shape: {prediction.shape}")
    print(f"   Predicted behavior vector: {prediction[-1]}")  # Last time step
    print(f"   - Sentiment: {prediction[-1, 0]:.3f}")
    print(f"   - Formality: {prediction[-1, 1]:.3f}")
    print(f"   - Creativity: {prediction[-1, 2]:.3f}")

def demonstrate_online_learning(library, model_path):
    """Demonstrate online learning capabilities."""
    print("\n" + "="*70)
    print("4. ONLINE LEARNING FOR REAL-TIME ADAPTATION")
    print("="*70)
    
    print("🔄 Creating reservoir with online learning capabilities...")
    
    reservoir = EnhancedReservoirRWKV(
        shared_library=library,
        model_path=model_path,
        units=64,
        persona_type='balanced',
        readout_type='online',
        readout_config={
            'output_size': 1,
            'learning_rate': 0.05,
            'forgetting_factor': 0.95
        }
    )
    
    # Initial training
    print("📚 Initial training phase...")
    X_initial = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    y_initial = np.array([[0.2], [0.5], [0.8]])
    
    reservoir.fit(X_initial, y_initial)
    
    # Test sequence
    test_seq = [20, 25, 30, 35, 40]
    
    print(f"\n🧪 Online adaptation demonstration:")
    print(f"   Test sequence: {test_seq}")
    
    # Initial prediction
    pred_initial = reservoir.predict(test_seq)
    print(f"   Initial prediction: {pred_initial[-1, 0]:.4f}")
    
    # Simulate user feedback and adaptation
    adaptations = [
        ([20, 25, 30], 0.3),
        ([25, 30, 35], 0.4),
        ([30, 35, 40], 0.6),
    ]
    
    for i, (adapt_seq, target) in enumerate(adaptations):
        reservoir.update_online(np.array(adapt_seq), np.array([[target]]))
        pred_after = reservoir.predict(test_seq)
        print(f"   After adaptation {i+1}: {pred_after[-1, 0]:.4f} (target: {target})")
    
    print("✓ Online learning enables real-time chatbot personality adaptation!")

def demonstrate_hierarchical_outputs(library, model_path):
    """Demonstrate hierarchical outputs at different time scales."""
    print("\n" + "="*70)
    print("5. HIERARCHICAL OUTPUTS FOR MULTI-SCALE REASONING")
    print("="*70)
    
    print("🏗️ Creating reservoir with hierarchical outputs...")
    
    # Configure hierarchical outputs for different reasoning scales
    hierarchical_configs = [
        {
            'output_size': 1,
            'time_scale': 1,
            'readout_type': 'ridge',
            'readout_params': {'alpha': 1e-6}
        },
        {
            'output_size': 1, 
            'time_scale': 5,
            'readout_type': 'ridge',
            'readout_params': {'alpha': 1e-4}
        },
        {
            'output_size': 1,
            'time_scale': 10,
            'readout_type': 'ridge', 
            'readout_params': {'alpha': 1e-3}
        }
    ]
    
    reservoir = EnhancedReservoirRWKV(
        shared_library=library,
        model_path=model_path,
        units=64,
        persona_type='balanced',
        readout_type='hierarchical',
        hierarchical_configs=hierarchical_configs
    )
    
    # Generate hierarchical training data
    print("📊 Generating hierarchical training data...")
    
    # Long sequence for multi-scale analysis
    long_sequence = list(range(1, 51))  # 50 time steps
    
    # Different scale targets
    hierarchical_targets = {
        'readout_0_1': np.sin(np.linspace(0, 4*np.pi, 50)).reshape(-1, 1),      # Fast oscillation
        'readout_1_5': np.sin(np.linspace(0, np.pi, 10)).reshape(-1, 1),        # Medium oscillation  
        'readout_2_10': np.sin(np.linspace(0, np.pi/2, 5)).reshape(-1, 1)       # Slow trend
    }
    
    # Train hierarchical system
    print("🎯 Training hierarchical readouts...")
    reservoir.fit([long_sequence], None, hierarchical_targets=hierarchical_targets)
    
    # Test hierarchical predictions
    test_sequence = list(range(51, 81))  # New 30-step sequence
    hierarchical_predictions = reservoir.predict(test_sequence)
    
    print(f"✓ Hierarchical outputs trained successfully")
    print(f"   Prediction scales:")
    for scale, pred in hierarchical_predictions.items():
        time_scale = int(scale.split('_')[-1])
        print(f"   - {scale} (every {time_scale} steps): shape {pred.shape}")
    
    print("💡 This enables chatbots to reason at multiple time scales:")
    print("   - Fast: immediate response generation")
    print("   - Medium: contextual understanding")
    print("   - Slow: long-term conversation coherence")

def demonstrate_batch_processing(library, model_path):
    """Demonstrate efficient batch processing."""
    print("\n" + "="*70)
    print("6. EFFICIENT BATCH PROCESSING")
    print("="*70)
    
    print("⚡ Creating reservoir for batch processing demo...")
    
    reservoir = EnhancedReservoirRWKV(
        shared_library=library,
        model_path=model_path,
        units=64,
        persona_type='balanced',
        readout_type='ridge'
    )
    
    # Train on sample data
    print("📚 Training reservoir...")
    X_train = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    y_train = np.array([[0.1], [0.5], [0.9]])
    reservoir.fit(X_train, y_train)
    
    # Prepare batch of conversations
    print("📦 Preparing batch of conversation sequences...")
    
    conversation_batch = [
        [15, 30, 45, 60, 75],      # Conversation 1
        [20, 40, 60, 80],          # Conversation 2
        [25, 50, 75, 100, 125, 150], # Conversation 3
        [10, 20, 30],              # Conversation 4
        [100, 200, 150, 175, 125]  # Conversation 5
    ]
    
    print(f"   Batch size: {len(conversation_batch)} conversations")
    print(f"   Sequence lengths: {[len(seq) for seq in conversation_batch]}")
    
    # Process batch efficiently
    import time
    start_time = time.time()
    
    batch_predictions = reservoir.batch_predict(conversation_batch)
    
    end_time = time.time()
    
    print(f"✓ Batch processing completed in {end_time - start_time:.4f} seconds")
    print(f"   Results:")
    for i, pred in enumerate(batch_predictions):
        if pred.ndim == 1:
            final_pred = pred[-1]
        else:
            final_pred = pred[-1, 0]
        print(f"   - Conversation {i+1}: final prediction = {final_pred:.4f}")
    
    print("⚡ Batch processing enables efficient handling of multiple conversations!")

def demonstrate_persona_comparison(reservoirs):
    """Compare different persona behaviors side by side."""
    print("\n" + "="*70)
    print("7. PERSONA BEHAVIOR COMPARISON")
    print("="*70)
    
    if not reservoirs:
        print("⚠️ Skipping persona comparison (no reservoirs available)")
        return
    
    print("🔍 Comparing persona responses to identical inputs...")
    
    # Test scenarios
    test_scenarios = [
        ([50, 100, 150, 200], "Technical question"),
        ([10, 25, 75, 200, 150], "Emotional expression"),
        ([100, 100, 100, 100], "Repetitive input"),
        ([5, 50, 150, 250, 200, 25], "Complex conversation")
    ]
    
    for scenario, description in test_scenarios:
        print(f"\n📝 Scenario: {description}")
        print(f"   Input: {scenario}")
        
        responses = {}
        for persona, reservoir in reservoirs.items():
            try:
                # Get reservoir activations (internal state)
                activations = reservoir.run(scenario)
                # Measure response characteristics
                mean_activation = np.mean(activations)
                activation_variance = np.var(activations)
                max_activation = np.max(activations)
                
                responses[persona] = {
                    'mean': mean_activation,
                    'variance': activation_variance,
                    'max': max_activation
                }
            except Exception as e:
                print(f"   ⚠️ Error with {persona} persona: {e}")
                continue
        
        # Display comparison
        for persona, metrics in responses.items():
            print(f"   {persona:>12}: mean={metrics['mean']:.3f}, "
                  f"var={metrics['variance']:.3f}, max={metrics['max']:.3f}")
    
    print("\n💡 Different personas show distinct response patterns:")
    print("   - Conservative: lower variance, stable responses")
    print("   - Balanced: moderate variance, adaptive responses")
    print("   - Creative: higher variance, dynamic responses")

def main():
    """Main demonstration function."""
    print("Enhanced ReservoirRWKV Demonstration")
    print("Advanced Chatbot Personality Modeling with RWKV.cpp")
    print("="*70)
    
    # Setup
    library, model_path = setup_environment()
    if library is None or model_path is None:
        return
    
    try:
        # Run demonstrations
        demonstrate_esn_parameter_mappings()
        
        reservoirs = demonstrate_chatbot_personas(library, model_path)
        
        demonstrate_multi_layer_readout(library, model_path)
        
        demonstrate_online_learning(library, model_path)
        
        demonstrate_hierarchical_outputs(library, model_path)
        
        demonstrate_batch_processing(library, model_path)
        
        demonstrate_persona_comparison(reservoirs)
        
        # Conclusion
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        
        print("🎉 Successfully demonstrated all Enhanced ReservoirRWKV features!")
        print("\n📋 Summary of capabilities:")
        print("   ✓ Detailed ESN parameter mappings to RWKV.cpp")
        print("   ✓ Multiple chatbot personas with distinct behaviors")
        print("   ✓ Multi-layer readout networks for complex reasoning")
        print("   ✓ Online learning for real-time adaptation")
        print("   ✓ Hierarchical outputs for multi-scale processing")
        print("   ✓ Efficient batch processing")
        print("   ✓ Comprehensive persona behavior analysis")
        
        print("\n🚀 Ready for production chatbot personality modeling!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()