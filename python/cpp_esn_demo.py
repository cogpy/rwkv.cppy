#!/usr/bin/env python3
"""
Comprehensive example demonstrating C++ ESN chatbot implementation.

This example showcases:
1. High-performance C++ ESN implementation
2. Python interface to C++ ESN
3. Chatbot personality modeling
4. Performance comparison with Python implementation
5. Real-time conversation capabilities
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rwkv_cpp import (
        RWKVSharedLibrary, 
        RWKVModel,
        ReservoirRWKV,  # Python implementation
        ESNRWKV,        # C++ implementation
        ESNPersonalityType,
        ESNReadoutType,
        create_chatbot_esn
    )
    print("✓ Successfully imported both Python and C++ ESN modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    print("Make sure you're running this from the correct directory and that both rwkv_cpp and esn are built.")
    sys.exit(1)

def setup_environment():
    """Set up the environment and check for required files."""
    # Find library and model
    rwkv_library_path = os.path.join(os.path.dirname(__file__), "..", "build", "librwkv.so")
    esn_library_path = os.path.join(os.path.dirname(__file__), "..", "build", "libesn.so") 
    test_model = os.path.join(os.path.dirname(__file__), "..", "tests", "tiny-rwkv-6v0-3m-FP32-to-Q8_0.bin")
    
    missing_files = []
    if not os.path.exists(rwkv_library_path):
        missing_files.append(f"RWKV library: {rwkv_library_path}")
    if not os.path.exists(esn_library_path):
        missing_files.append(f"ESN library: {esn_library_path}")
    if not os.path.exists(test_model):
        missing_files.append(f"Test model: {test_model}")
    
    if missing_files:
        print("✗ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please build the project first: cmake -B build . && cd build && make")
        return None, None, None
    
    return rwkv_library_path, esn_library_path, test_model

def demonstrate_cpp_esn():
    """Demonstrate the C++ ESN implementation."""
    print("\n" + "="*60)
    print("C++ ESN IMPLEMENTATION DEMONSTRATION")
    print("="*60)
    
    rwkv_lib_path, esn_lib_path, model_path = setup_environment()
    if not all([rwkv_lib_path, esn_lib_path, model_path]):
        return False
    
    try:
        # Initialize RWKV
        print("\n1. Initializing RWKV backend...")
        rwkv_lib = RWKVSharedLibrary(rwkv_lib_path)
        rwkv_model = RWKVModel(rwkv_lib, model_path, n_threads=4)
        print(f"✓ RWKV model loaded: {model_path}")
        
        # Test different personalities
        personalities = [
            (ESNPersonalityType.CONSERVATIVE, "Conservative (Stable)"),
            (ESNPersonalityType.BALANCED, "Balanced (Adaptive)"),
            (ESNPersonalityType.CREATIVE, "Creative (Dynamic)")
        ]
        
        for personality_type, personality_name in personalities:
            print(f"\n2. Testing {personality_name} Personality...")
            
            # Create C++ ESN
            start_time = time.time()
            esn = ESNRWKV(
                rwkv_model=rwkv_model,
                esn_library_path=esn_lib_path,
                personality=personality_type,
                units=128,  # Smaller for faster demo
                warmup_steps=5
            )
            init_time = time.time() - start_time
            
            print(f"✓ C++ ESN initialized in {init_time:.3f}s")
            print(f"  Reservoir size: {esn.get_reservoir_size()}")
            print(f"  Personality: {esn.get_personality().name}")
            
            # Test reservoir operation
            test_tokens = [1, 5, 10, 15, 20, 25, 30]
            print(f"\n3. Running reservoir on test sequence: {test_tokens}")
            
            start_time = time.time()
            activations = esn.run(test_tokens)
            run_time = time.time() - start_time
            
            print(f"✓ Reservoir run completed in {run_time:.4f}s")
            print(f"  Activation shape: {activations.shape}")
            print(f"  Activation stats: mean={np.mean(activations):.4f}, std={np.std(activations):.4f}")
            
            # Test conversation functionality
            print(f"\n4. Testing conversation features...")
            esn.init_conversation()
            print("✓ Conversation state initialized")
            
            # Test personality switching
            if personality_type != ESNPersonalityType.CREATIVE:
                switch_success = esn.switch_personality(ESNPersonalityType.CREATIVE)
                if switch_success:
                    print("✓ Successfully switched to creative personality")
                    new_personality = esn.get_personality()
                    print(f"  Current personality: {new_personality.name}")
            
            # Reset for next test
            esn.reset_state()
            del esn
            
        print("\n✓ All personality tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during C++ ESN demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_performance_comparison():
    """Compare Python vs C++ ESN performance."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON: PYTHON vs C++")
    print("="*60)
    
    rwkv_lib_path, esn_lib_path, model_path = setup_environment()
    if not all([rwkv_lib_path, esn_lib_path, model_path]):
        return False
    
    try:
        # Initialize RWKV
        rwkv_lib = RWKVSharedLibrary(rwkv_lib_path)
        rwkv_model = RWKVModel(rwkv_lib, model_path, n_threads=4)
        
        # Test parameters
        test_sequences = [
            [1, 2, 3, 4, 5],
            [10, 15, 20, 25, 30, 35],
            [50, 55, 60, 65, 70, 75, 80, 85],
            list(range(100, 120))  # Longer sequence
        ]
        
        print(f"Testing on {len(test_sequences)} sequences of varying lengths...")
        
        # Initialize Python ESN
        print("\n1. Python ESN (ReservoirRWKV)...")
        python_esn = ReservoirRWKV(
            shared_library=rwkv_lib,
            model_path=model_path,
            units=128,
            alpha=1e-4
        )
        
        # Initialize C++ ESN
        print("2. C++ ESN (ESNRWKV)...")
        cpp_esn = ESNRWKV(
            rwkv_model=rwkv_model,
            esn_library_path=esn_lib_path,
            personality=ESNPersonalityType.BALANCED,
            units=128
        )
        
        # Performance testing
        print("\n3. Running performance tests...")
        python_times = []
        cpp_times = []
        
        for i, sequence in enumerate(test_sequences):
            print(f"\n  Sequence {i+1} (length {len(sequence)}): {sequence[:5]}{'...' if len(sequence) > 5 else ''}")
            
            # Test Python implementation
            try:
                start_time = time.time()
                python_activations = python_esn.run(sequence)
                python_time = time.time() - start_time
                python_times.append(python_time)
                print(f"    Python: {python_time:.4f}s, shape: {python_activations.shape}")
            except Exception as e:
                print(f"    Python: Failed ({e})")
                python_times.append(float('inf'))
            
            # Test C++ implementation
            try:
                start_time = time.time()
                cpp_activations = cpp_esn.run(sequence)
                cpp_time = time.time() - start_time
                cpp_times.append(cpp_time)
                print(f"    C++:    {cpp_time:.4f}s, shape: {cpp_activations.shape}")
            except Exception as e:
                print(f"    C++: Failed ({e})")
                cpp_times.append(float('inf'))
        
        # Summary
        print("\n4. Performance Summary:")
        valid_comparisons = [(p, c) for p, c in zip(python_times, cpp_times) 
                           if p != float('inf') and c != float('inf')]
        
        if valid_comparisons:
            avg_python = np.mean([p for p, c in valid_comparisons])
            avg_cpp = np.mean([c for p, c in valid_comparisons])
            speedup = avg_python / avg_cpp if avg_cpp > 0 else float('inf')
            
            print(f"  Average Python time: {avg_python:.4f}s")
            print(f"  Average C++ time:    {avg_cpp:.4f}s")
            print(f"  Speedup factor:      {speedup:.2f}x")
            
            if speedup > 1:
                print("✓ C++ implementation is faster!")
            elif speedup < 1:
                print("⚠ Python implementation is faster (unexpected)")
            else:
                print("≈ Similar performance")
        else:
            print("✗ No valid comparisons could be made")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during performance comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_chatbot_interaction():
    """Demonstrate interactive chatbot functionality."""
    print("\n" + "="*60) 
    print("INTERACTIVE CHATBOT DEMONSTRATION")
    print("="*60)
    
    rwkv_lib_path, esn_lib_path, model_path = setup_environment()
    if not all([rwkv_lib_path, esn_lib_path, model_path]):
        return False
    
    try:
        # Initialize RWKV and ESN
        rwkv_lib = RWKVSharedLibrary(rwkv_lib_path)
        rwkv_model = RWKVModel(rwkv_lib, model_path, n_threads=4)
        
        # Create chatbot with different personalities
        chatbots = {}
        personalities = [
            (ESNPersonalityType.CONSERVATIVE, "Conservative"),
            (ESNPersonalityType.BALANCED, "Balanced"),
            (ESNPersonalityType.CREATIVE, "Creative")
        ]
        
        print("\n1. Initializing chatbots with different personalities...")
        for personality_type, name in personalities:
            chatbot = create_chatbot_esn(rwkv_model, esn_lib_path, personality_type)
            chatbots[name] = chatbot
            print(f"✓ {name} chatbot ready (reservoir size: {chatbot.get_reservoir_size()})")
        
        # Simulate conversation inputs
        conversation_inputs = [
            "Hello, how are you?",
            "What's the weather like?", 
            "Tell me a joke",
            "What do you think about AI?",
            "Goodbye"
        ]
        
        print(f"\n2. Simulating conversation with {len(conversation_inputs)} inputs...")
        
        # Convert text to simple token sequences (simplified tokenization)
        def simple_tokenize(text: str) -> List[int]:
            """Simple tokenization for demonstration."""
            return [hash(word) % 1000 + 1 for word in text.lower().split()]
        
        for i, input_text in enumerate(conversation_inputs):
            print(f"\n  Turn {i+1}: '{input_text}'")
            tokens = simple_tokenize(input_text)
            print(f"    Tokens: {tokens}")
            
            for name, chatbot in chatbots.items():
                try:
                    # Get reservoir response (representing internal processing)
                    start_time = time.time()
                    activations = chatbot.run(tokens, reset_state=False)  # Keep conversation state
                    process_time = time.time() - start_time
                    
                    # Calculate response characteristics
                    response_energy = np.mean(np.abs(activations))
                    response_variability = np.std(activations)
                    
                    print(f"    {name:12} | Time: {process_time:.3f}s | "
                          f"Energy: {response_energy:.3f} | Var: {response_variability:.3f}")
                    
                except Exception as e:
                    print(f"    {name:12} | Error: {e}")
        
        # Test personality switching
        print("\n3. Testing personality switching...")
        test_chatbot = chatbots["Balanced"]
        test_input = simple_tokenize("Tell me about yourself")
        
        original_response = test_chatbot.run(test_input)
        original_energy = np.mean(np.abs(original_response))
        print(f"  Original (Balanced): Energy = {original_energy:.3f}")
        
        # Switch to creative
        test_chatbot.switch_personality(ESNPersonalityType.CREATIVE)
        creative_response = test_chatbot.run(test_input)
        creative_energy = np.mean(np.abs(creative_response))
        print(f"  After switch (Creative): Energy = {creative_energy:.3f}")
        
        energy_change = (creative_energy - original_energy) / original_energy * 100
        print(f"  Energy change: {energy_change:+.1f}%")
        
        if abs(energy_change) > 5:
            print("✓ Personality switch had measurable effect on responses")
        else:
            print("~ Personality switch had minimal effect (may need tuning)")
        
        print("\n✓ Chatbot interaction demonstration completed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during chatbot demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demonstration function."""
    print("C++ ESN CHATBOT COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the high-performance C++ ESN implementation")
    print("with Python bindings for advanced chatbot personality modeling.")
    
    # Run all demonstrations
    demos = [
        ("C++ ESN Implementation", demonstrate_cpp_esn),
        ("Performance Comparison", demonstrate_performance_comparison),
        ("Chatbot Interaction", demonstrate_chatbot_interaction)
    ]
    
    results = {}
    for name, demo_func in demos:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            results[name] = demo_func()
        except Exception as e:
            print(f"✗ Demo '{name}' failed with exception: {e}")
            results[name] = False
    
    # Final summary
    print("\n" + "="*60)
    print("DEMONSTRATION SUMMARY")
    print("="*60)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:30} | {status}")
    
    total_passed = sum(results.values())
    total_demos = len(results)
    print(f"\nOverall: {total_passed}/{total_demos} demonstrations passed")
    
    if total_passed == total_demos:
        print("\n🎉 All demonstrations completed successfully!")
        print("The C++ ESN implementation is working correctly.")
    else:
        print(f"\n⚠ {total_demos - total_passed} demonstration(s) failed.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()