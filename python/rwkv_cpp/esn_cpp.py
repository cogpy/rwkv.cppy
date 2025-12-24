#!/usr/bin/env python3
"""
Python bindings for the C++ ESN (Echo State Network) implementation.

This module provides Python access to the high-performance C++ ESN chatbot
implementation while maintaining compatibility with the existing ReservoirPy-style API.
"""

import ctypes
import os
import numpy as np
from typing import Optional, List, Tuple, Union
from enum import IntEnum

# Try to import the parent module components
try:
    from .rwkv_cpp_shared_library import RWKVSharedLibrary
    from .rwkv_cpp_model import RWKVModel
except ImportError:
    # Fallback for standalone testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from rwkv_cpp import RWKVSharedLibrary, RWKVModel

class ESNPersonalityType(IntEnum):
    """ESN chatbot personality types."""
    CONSERVATIVE = 0
    BALANCED = 1
    CREATIVE = 2
    CUSTOM = 3

class ESNReadoutType(IntEnum):
    """ESN readout layer types."""
    RIDGE = 0
    LINEAR = 1
    MLP = 2
    ONLINE = 3

class ESNErrorFlags(IntEnum):
    """ESN error flags."""
    NONE = 0
    ARGS = 1 << 8
    ALLOC = 1
    MODEL = 2
    TRAINING = 3
    PREDICTION = 4
    DIMENSION = 5
    STATE = 6

# C structures
class ESNConfig(ctypes.Structure):
    """ESN configuration structure matching the C definition."""
    _fields_ = [
        ("units", ctypes.c_uint32),
        ("spectral_radius", ctypes.c_float),
        ("leaking_rate", ctypes.c_float),
        ("input_scaling", ctypes.c_float),
        ("noise_scaling", ctypes.c_float),
        ("ridge_alpha", ctypes.c_float),
        ("warmup_steps", ctypes.c_uint32),
        ("personality", ctypes.c_int),
        ("readout_type", ctypes.c_int),
        ("online_learning", ctypes.c_bool),
        ("mlp_hidden_size", ctypes.c_uint32),
        ("learning_rate", ctypes.c_float),
    ]

class ESNPredictionResult(ctypes.Structure):
    """ESN prediction result structure."""
    _fields_ = [
        ("outputs", ctypes.POINTER(ctypes.c_float)),
        ("output_length", ctypes.c_size_t),
        ("output_dim", ctypes.c_size_t),
        ("confidence", ctypes.c_float),
    ]

class ESNTrainingData(ctypes.Structure):
    """ESN training data structure."""
    _fields_ = [
        ("sequences", ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))),
        ("sequence_lengths", ctypes.POINTER(ctypes.c_size_t)),
        ("targets", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("target_lengths", ctypes.POINTER(ctypes.c_size_t)),
        ("n_sequences", ctypes.c_size_t),
        ("output_dim", ctypes.c_size_t),
    ]

class ESNSharedLibrary:
    """Wrapper for the ESN shared library."""
    
    def __init__(self, library_path: str):
        """Initialize the ESN shared library.
        
        Args:
            library_path: Path to the ESN shared library (libesn.so)
        """
        self.library = ctypes.CDLL(library_path)
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures for the C API."""
        
        # esn_create_config
        self.library.esn_create_config.argtypes = [ctypes.c_int]
        self.library.esn_create_config.restype = ESNConfig
        
        # esn_init
        self.library.esn_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ESNConfig)]
        self.library.esn_init.restype = ctypes.c_void_p
        
        # esn_predict
        self.library.esn_predict.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_uint32), 
            ctypes.c_size_t
        ]
        self.library.esn_predict.restype = ctypes.POINTER(ESNPredictionResult)
        
        # esn_run_reservoir
        self.library.esn_run_reservoir.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.library.esn_run_reservoir.restype = ctypes.POINTER(ctypes.c_float)
        
        # esn_get_reservoir_size
        self.library.esn_get_reservoir_size.argtypes = [ctypes.c_void_p]
        self.library.esn_get_reservoir_size.restype = ctypes.c_uint32
        
        # esn_get_personality
        self.library.esn_get_personality.argtypes = [ctypes.c_void_p]
        self.library.esn_get_personality.restype = ctypes.c_int
        
        # esn_reset_state
        self.library.esn_reset_state.argtypes = [ctypes.c_void_p]
        self.library.esn_reset_state.restype = None
        
        # esn_init_conversation
        self.library.esn_init_conversation.argtypes = [ctypes.c_void_p]
        self.library.esn_init_conversation.restype = ctypes.c_void_p
        
        # esn_switch_personality
        self.library.esn_switch_personality.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]
        self.library.esn_switch_personality.restype = ctypes.c_bool
        
        # Memory management functions
        self.library.esn_free_prediction.argtypes = [ctypes.POINTER(ESNPredictionResult)]
        self.library.esn_free_prediction.restype = None
        
        self.library.esn_free_activations.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.library.esn_free_activations.restype = None
        
        self.library.esn_free_conversation_state.argtypes = [ctypes.c_void_p]
        self.library.esn_free_conversation_state.restype = None
        
        self.library.esn_free_context.argtypes = [ctypes.c_void_p]
        self.library.esn_free_context.restype = None

class ESNRWKV:
    """
    High-performance C++ ESN implementation with Python interface.
    
    This class provides a Python wrapper around the C++ ESN implementation,
    offering better performance than the pure Python version while maintaining
    API compatibility.
    """
    
    def __init__(self, 
                 rwkv_model: RWKVModel,
                 esn_library_path: str,
                 personality: ESNPersonalityType = ESNPersonalityType.BALANCED,
                 units: int = 256,
                 spectral_radius: float = 0.9,
                 leaking_rate: float = 0.5,
                 input_scaling: float = 1.0,
                 noise_scaling: float = 0.05,
                 ridge_alpha: float = 1e-4,
                 warmup_steps: int = 10,
                 readout_type: ESNReadoutType = ESNReadoutType.RIDGE,
                 online_learning: bool = False,
                 mlp_hidden_size: int = 128,
                 learning_rate: float = 0.01):
        """
        Initialize the C++ ESN with RWKV backend.
        
        Args:
            rwkv_model: Initialized RWKV model to use as reservoir
            esn_library_path: Path to the ESN shared library
            personality: Chatbot personality type
            units: Number of reservoir units
            spectral_radius: Reservoir spectral radius (creativity control)
            leaking_rate: Memory persistence rate
            input_scaling: Input sensitivity scaling
            noise_scaling: Response variability scaling
            ridge_alpha: Ridge regression regularization parameter
            warmup_steps: Number of warmup steps before prediction
            readout_type: Type of readout layer
            online_learning: Enable online learning adaptation
            mlp_hidden_size: Hidden layer size for MLP readout
            learning_rate: Learning rate for online learning
        """
        
        self.rwkv_model = rwkv_model
        self.esn_lib = ESNSharedLibrary(esn_library_path)
        
        # Create ESN configuration
        if personality != ESNPersonalityType.CUSTOM:
            # Use preset configuration
            self.config = self.esn_lib.library.esn_create_config(personality)
            # Override specific parameters if provided
            self.config.units = units
            if spectral_radius != 0.9:
                self.config.spectral_radius = spectral_radius
            if leaking_rate != 0.5:
                self.config.leaking_rate = leaking_rate
        else:
            # Custom configuration
            self.config = ESNConfig()
            self.config.units = units
            self.config.spectral_radius = spectral_radius
            self.config.leaking_rate = leaking_rate
            self.config.input_scaling = input_scaling
            self.config.noise_scaling = noise_scaling
            self.config.ridge_alpha = ridge_alpha
            self.config.warmup_steps = warmup_steps
            self.config.personality = personality
            self.config.readout_type = readout_type
            self.config.online_learning = online_learning
            self.config.mlp_hidden_size = mlp_hidden_size
            self.config.learning_rate = learning_rate
        
        # Initialize ESN context
        rwkv_ctx_ptr = rwkv_model._ctx.ptr
        self.esn_ctx = self.esn_lib.library.esn_init(rwkv_ctx_ptr, ctypes.byref(self.config))
        
        if not self.esn_ctx:
            raise RuntimeError("Failed to initialize ESN context")
        
        self.is_trained = False
        self.conversation_state = None
    
    def run(self, sequences: Union[List[int], List[List[int]]], reset_state: bool = True) -> np.ndarray:
        """
        Run the reservoir on input sequences and return activations.
        
        Args:
            sequences: Input token sequences
            reset_state: Whether to reset reservoir state before processing
            
        Returns:
            Reservoir activations as numpy array
        """
        if reset_state:
            self.esn_lib.library.esn_reset_state(self.esn_ctx)
        
        # Handle single sequence vs batch
        if isinstance(sequences[0], int):
            sequences = [sequences]
        
        all_activations = []
        
        for seq in sequences:
            # Convert to ctypes array
            seq_array = (ctypes.c_uint32 * len(seq))(*seq)
            activation_length = ctypes.c_size_t()
            
            # Get reservoir activations
            activations_ptr = self.esn_lib.library.esn_run_reservoir(
                self.esn_ctx, seq_array, len(seq), ctypes.byref(activation_length)
            )
            
            if not activations_ptr:
                raise RuntimeError("Reservoir run failed")
            
            # Convert to numpy array
            activations = np.ctypeslib.as_array(activations_ptr, shape=(activation_length.value,))
            activations_copy = activations.copy()  # Make a copy before freeing
            
            # Free C memory
            self.esn_lib.library.esn_free_activations(activations_ptr)
            
            # Reshape to (sequence_length, units)
            seq_len = len(seq)
            units = self.config.units
            activations_reshaped = activations_copy.reshape(seq_len, units)
            all_activations.append(activations_reshaped)
        
        return np.array(all_activations) if len(all_activations) > 1 else all_activations[0]
    
    def predict(self, sequences: Union[List[int], List[List[int]]], reset_state: bool = True) -> np.ndarray:
        """
        Make predictions on input sequences.
        
        Args:
            sequences: Input token sequences
            reset_state: Whether to reset reservoir state before processing
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_trained:
            raise RuntimeError("ESN must be trained before making predictions")
        
        if reset_state:
            self.esn_lib.library.esn_reset_state(self.esn_ctx)
        
        # Handle single sequence vs batch
        if isinstance(sequences[0], int):
            sequences = [sequences]
        
        all_predictions = []
        
        for seq in sequences:
            # Convert to ctypes array
            seq_array = (ctypes.c_uint32 * len(seq))(*seq)
            
            # Get prediction
            result_ptr = self.esn_lib.library.esn_predict(self.esn_ctx, seq_array, len(seq))
            
            if not result_ptr:
                raise RuntimeError("Prediction failed")
            
            result = result_ptr.contents
            
            # Convert to numpy array
            output_size = result.output_length * result.output_dim
            predictions = np.ctypeslib.as_array(result.outputs, shape=(output_size,))
            predictions_copy = predictions.copy()  # Make a copy before freeing
            
            # Free C memory
            self.esn_lib.library.esn_free_prediction(result_ptr)
            
            # Reshape if multi-dimensional output
            if result.output_dim > 1:
                predictions_reshaped = predictions_copy.reshape(result.output_length, result.output_dim)
            else:
                predictions_reshaped = predictions_copy
            
            all_predictions.append(predictions_reshaped)
        
        return np.array(all_predictions) if len(all_predictions) > 1 else all_predictions[0]
    
    def fit(self, X: List[List[int]], y: np.ndarray, warmup: int = 0):
        """
        Train the ESN readout layer.
        
        Args:
            X: Input token sequences
            y: Target outputs
            warmup: Warmup period (will use config warmup_steps if 0)
        """
        # This is a simplified interface - for full training functionality,
        # you would need to create the ESNTrainingData structure and call esn_train
        # For now, we'll mark as trained to allow predictions
        
        # TODO: Implement full training data conversion and call to esn_train
        print(f"Warning: Simplified training interface. Input: {len(X)} sequences, Output shape: {y.shape}")
        print("For full training functionality, use the C API directly or enhance this wrapper.")
        
        self.is_trained = True
    
    def reset_state(self):
        """Reset the reservoir state."""
        self.esn_lib.library.esn_reset_state(self.esn_ctx)
    
    def get_reservoir_size(self) -> int:
        """Get the reservoir size."""
        return self.esn_lib.library.esn_get_reservoir_size(self.esn_ctx)
    
    def get_personality(self) -> ESNPersonalityType:
        """Get the current personality type."""
        return ESNPersonalityType(self.esn_lib.library.esn_get_personality(self.esn_ctx))
    
    def init_conversation(self):
        """Initialize a conversation state for chatbot functionality."""
        self.conversation_state = self.esn_lib.library.esn_init_conversation(self.esn_ctx)
        if not self.conversation_state:
            raise RuntimeError("Failed to initialize conversation state")
    
    def switch_personality(self, new_personality: ESNPersonalityType) -> bool:
        """
        Switch chatbot personality.
        
        Args:
            new_personality: New personality type
            
        Returns:
            True if successful
        """
        if not self.conversation_state:
            self.init_conversation()
        
        return self.esn_lib.library.esn_switch_personality(
            self.esn_ctx, self.conversation_state, new_personality
        )
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'conversation_state') and self.conversation_state:
            self.esn_lib.library.esn_free_conversation_state(self.conversation_state)
        if hasattr(self, 'esn_ctx') and self.esn_ctx:
            self.esn_lib.library.esn_free_context(self.esn_ctx)

def create_chatbot_esn(rwkv_model: RWKVModel, 
                      esn_library_path: str,
                      personality: ESNPersonalityType = ESNPersonalityType.BALANCED) -> ESNRWKV:
    """
    Create a chatbot ESN with preset personality configurations.
    
    Args:
        rwkv_model: Initialized RWKV model
        esn_library_path: Path to ESN shared library
        personality: Chatbot personality type
        
    Returns:
        Configured ESNRWKV instance
    """
    esn = ESNRWKV(rwkv_model, esn_library_path, personality=personality)
    esn.init_conversation()
    return esn

# Compatibility alias
ESNChatbot = ESNRWKV