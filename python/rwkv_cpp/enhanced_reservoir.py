"""
Enhanced ReservoirPy-compatible implementation using RWKV as the reservoir layer.

This module provides a comprehensive Echo State Network implementation with detailed
mapping of ReservoirPy parameters to RWKV.cpp equivalents, designed for chatbot 
personality modeling and advanced reservoir computing tasks.

Key Features:
- Detailed ESN parameter mapping from ReservoirPy to RWKV.cpp
- Multi-layer readout networks  
- Online learning capabilities
- Hierarchical outputs at different time scales
- Custom readout layer support
- Batch processing
- Chatbot persona modeling features
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any, Dict, Callable
from .rwkv_cpp_model import RWKVModel
from .rwkv_cpp_shared_library import RWKVSharedLibrary
from .reservoir import ReservoirRWKV
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import Ridge, SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import reservoirpy as rpy
    HAS_RESERVOIRPY = True
except ImportError:
    HAS_RESERVOIRPY = False


class ESNParameterMapping:
    """
    Detailed mapping of ReservoirPy ESN parameters to RWKV.cpp equivalents
    for chatbot personality modeling.
    """
    
    @staticmethod
    def get_parameter_mappings() -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive mapping of ReservoirPy parameters to RWKV.cpp concepts.
        
        Returns mapping for chatbot personality modeling where:
        - Spectral radius affects response stability/creativity
        - Leaking rate affects memory persistence 
        - Input scaling affects sensitivity to inputs
        - Density affects feature interaction complexity
        """
        return {
            'spectral_radius': {
                'reservoirpy_description': 'Largest eigenvalue of reservoir weight matrix',
                'rwkv_equivalent': 'Layer normalization scaling in RWKV layers',
                'rwkv_parameter': 'ln_weight and ln_bias scaling factors',
                'chatbot_persona_effect': 'Controls response stability vs creativity',
                'implementation': 'Adjust hidden state scaling after RWKV forward pass',
                'value_range': (0.1, 1.5),
                'default_value': 0.9,
                'personality_mapping': {
                    'conservative': 0.7,   # Stable, predictable responses
                    'balanced': 0.9,       # Standard behavior
                    'creative': 1.2        # More dynamic, creative responses
                }
            },
            
            'leaking_rate': {
                'reservoirpy_description': 'Rate of state decay (1 = no decay, 0 = instant decay)',
                'rwkv_equivalent': 'Exponential moving average in time-mixing',
                'rwkv_parameter': 'time_mix_k, time_mix_v, time_mix_r values',
                'chatbot_persona_effect': 'Controls memory persistence and context retention',
                'implementation': 'Modify time-mixing coefficients in RWKV attention',
                'value_range': (0.1, 1.0),
                'default_value': 1.0,
                'personality_mapping': {
                    'forgetful': 0.3,      # Short memory, focus on recent context
                    'balanced': 0.7,       # Standard memory retention
                    'long_memory': 0.95    # Extended context retention
                }
            },
            
            'input_scaling': {
                'reservoirpy_description': 'Scaling factor for input signals',
                'rwkv_equivalent': 'Token embedding scaling and input preprocessing',
                'rwkv_parameter': 'Embedding matrix normalization and input scaling',
                'chatbot_persona_effect': 'Controls sensitivity to user inputs',
                'implementation': 'Scale input embeddings before RWKV processing',
                'value_range': (0.1, 2.0),
                'default_value': 1.0,
                'personality_mapping': {
                    'subtle': 0.5,         # Less reactive to input variations
                    'balanced': 1.0,       # Standard sensitivity
                    'sensitive': 1.5       # Highly reactive to input nuances
                }
            },
            
            'density': {
                'reservoirpy_description': 'Connectivity density of reservoir matrix',
                'rwkv_equivalent': 'Channel mixing density and connection patterns',
                'rwkv_parameter': 'Channel-mixing layer connectivity patterns',
                'chatbot_persona_effect': 'Controls feature interaction complexity',
                'implementation': 'Selective activation of RWKV channels/features',
                'value_range': (0.1, 1.0),
                'default_value': 0.1,
                'personality_mapping': {
                    'focused': 0.05,       # Sparse, focused feature interactions
                    'balanced': 0.1,       # Standard feature connectivity
                    'complex': 0.3         # Rich feature interactions
                }
            },
            
            'bias_scaling': {
                'reservoirpy_description': 'Scaling factor for reservoir bias terms',
                'rwkv_equivalent': 'Layer bias terms in RWKV layers',
                'rwkv_parameter': 'Bias terms in attention and channel mixing',
                'chatbot_persona_effect': 'Controls baseline activation levels',
                'implementation': 'Adjust bias terms in RWKV layer computations',
                'value_range': (0.0, 1.0),
                'default_value': 0.0,
                'personality_mapping': {
                    'neutral': 0.0,        # No bias adjustment
                    'positive': 0.3,       # Positive bias for optimistic responses
                    'dynamic': 0.1         # Slight bias for dynamic behavior
                }
            },
            
            'noise_scaling': {
                'reservoirpy_description': 'Gaussian noise added to reservoir states',
                'rwkv_equivalent': 'Dropout and regularization in RWKV',
                'rwkv_parameter': 'Dropout probability and noise injection',
                'chatbot_persona_effect': 'Controls response variability and creativity',
                'implementation': 'Add controlled noise to RWKV hidden states',
                'value_range': (0.0, 0.1),
                'default_value': 0.0,
                'personality_mapping': {
                    'deterministic': 0.0,   # Consistent, predictable responses
                    'varied': 0.01,         # Slight variation in responses
                    'creative': 0.05        # Higher variability for creativity
                }
            }
        }


class MultiLayerReadout:
    """
    Multi-layer neural network readout for complex output mapping.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_torch: bool = True
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.use_torch = use_torch
        self.is_trained = False
        
        if use_torch and HAS_TORCH:
            self._build_torch_model()
        elif HAS_SKLEARN:
            self._build_sklearn_model()
        else:
            raise ImportError("Either PyTorch or scikit-learn is required")
    
    def _build_torch_model(self):
        """Build PyTorch neural network model."""
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
    
    def _build_sklearn_model(self):
        """Build scikit-learn MLP model."""
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layers),
            activation=self.activation,
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the readout model."""
        if self.use_torch and HAS_TORCH:
            self._fit_torch(X, y, epochs)
        else:
            self._fit_sklearn(X, y)
        self.is_trained = True
    
    def _fit_torch(self, X: np.ndarray, y: np.ndarray, epochs: int):
        """Train PyTorch model."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
    
    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray):
        """Train scikit-learn model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        if self.use_torch and HAS_TORCH:
            return self._predict_torch(X)
        else:
            return self._predict_sklearn(X)
    
    def _predict_torch(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with PyTorch model."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            return outputs.numpy()
    
    def _predict_sklearn(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with scikit-learn model."""
        return self.model.predict(X)


class OnlineLearner:
    """
    Online learning capability for incremental weight updates.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        forgetting_factor: float = 0.99
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
        
        self.is_initialized = True
    
    def update(self, x: np.ndarray, y: np.ndarray):
        """
        Update weights using simple gradient descent (simpler than RLS).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        # Simple gradient descent update
        for i in range(x.shape[0]):
            xi = x[i:i+1]  # (1, n_features)
            yi = y[i:i+1]  # (1, n_outputs)
            
            # Forward pass
            prediction = xi @ self.weights + self.bias  # (1, n_outputs)
            error = yi - prediction  # (1, n_outputs)
            
            # Gradient descent update
            # dW = learning_rate * xi.T @ error
            self.weights += self.learning_rate * xi.T @ error  # (n_features, n_outputs)
            self.bias += self.learning_rate * error.flatten()  # (n_outputs,)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions with current weights."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x @ self.weights + self.bias


class HierarchicalOutput:
    """
    Multiple readout layers operating at different temporal scales.
    """
    
    def __init__(
        self,
        input_size: int,
        output_configs: List[Dict[str, Any]]
    ):
        """
        Initialize hierarchical outputs.
        
        output_configs: List of dicts with keys:
        - 'output_size': int, number of outputs
        - 'time_scale': int, temporal downsampling factor
        - 'readout_type': str, type of readout ('ridge', 'mlp', 'online')
        - 'readout_params': dict, parameters for the readout
        """
        self.input_size = input_size
        self.output_configs = output_configs
        self.readouts = {}
        
        for i, config in enumerate(output_configs):
            readout_id = f"readout_{i}_{config['time_scale']}"
            
            if config['readout_type'] == 'ridge':
                from sklearn.linear_model import Ridge
                readout = Ridge(**config.get('readout_params', {}))
            elif config['readout_type'] == 'mlp':
                readout = MultiLayerReadout(
                    input_size=input_size,
                    output_size=config['output_size'],
                    **config.get('readout_params', {})
                )
            elif config['readout_type'] == 'online':
                readout = OnlineLearner(
                    input_size=input_size,
                    output_size=config['output_size'],
                    **config.get('readout_params', {})
                )
            else:
                raise ValueError(f"Unknown readout type: {config['readout_type']}")
            
            self.readouts[readout_id] = {
                'model': readout,
                'config': config,
                'is_trained': False
            }
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """
        Train all readout layers.
        
        y_dict: Dictionary mapping readout_id to target arrays
        """
        for readout_id, readout_info in self.readouts.items():
            if readout_id not in y_dict:
                continue
            
            config = readout_info['config']
            time_scale = config['time_scale']
            
            # Downsample features according to time scale
            X_downsampled = X[::time_scale]
            y_targets = y_dict[readout_id]
            
            # Train the readout
            model = readout_info['model']
            if hasattr(model, 'fit'):
                model.fit(X_downsampled, y_targets)
            else:
                # For online learners, train incrementally
                for i in range(min(len(X_downsampled), len(y_targets))):
                    model.update(X_downsampled[i:i+1], y_targets[i:i+1])
            
            readout_info['is_trained'] = True
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions with all readout layers."""
        predictions = {}
        
        for readout_id, readout_info in self.readouts.items():
            if not readout_info['is_trained']:
                continue
            
            config = readout_info['config']
            time_scale = config['time_scale']
            
            # Downsample features
            X_downsampled = X[::time_scale]
            
            # Make predictions
            model = readout_info['model']
            pred = model.predict(X_downsampled)
            predictions[readout_id] = pred
        
        return predictions


class EnhancedReservoirRWKV(ReservoirRWKV):
    """
    Enhanced ReservoirRWKV with detailed ESN parameter mapping and advanced features.
    
    Provides comprehensive reservoir computing capabilities with:
    - Detailed ReservoirPy parameter mapping
    - Multi-layer readout networks
    - Online learning
    - Hierarchical outputs  
    - Custom readout support
    - Batch processing
    - Chatbot persona modeling
    """
    
    def __init__(
        self,
        shared_library: RWKVSharedLibrary,
        model_path: str,
        
        # Basic reservoir parameters
        units: Optional[int] = None,
        
        # ESN parameter mappings (ReservoirPy equivalent)
        spectral_radius: float = 0.9,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        density: float = 0.1,
        bias_scaling: float = 0.0,
        noise_scaling: float = 0.0,
        
        # Chatbot persona configuration
        persona_type: str = 'balanced',  # 'conservative', 'balanced', 'creative', etc.
        
        # Readout configuration
        readout_type: str = 'ridge',  # 'ridge', 'mlp', 'online', 'hierarchical'
        readout_config: Optional[Dict[str, Any]] = None,
        
        # Advanced features
        enable_online_learning: bool = False,
        enable_hierarchical_output: bool = False,
        hierarchical_configs: Optional[List[Dict[str, Any]]] = None,
        
        # Standard RWKV parameters
        thread_count: Optional[int] = None,
        gpu_layer_count: int = 0,
        use_numpy: bool = True,
        dtype: Any = np.float32,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            shared_library=shared_library,
            model_path=model_path,
            units=units,
            thread_count=thread_count,
            gpu_layer_count=gpu_layer_count,
            use_numpy=use_numpy,
            dtype=dtype,
            **kwargs
        )
        
        # Store ESN parameters
        self.esn_params = ESNParameterMapping.get_parameter_mappings()
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.density = density
        self.bias_scaling = bias_scaling
        self.noise_scaling = noise_scaling
        
        # Apply persona configuration
        self.persona_type = persona_type
        self._apply_persona_configuration()
        
        # Initialize advanced readout systems
        self.readout_type = readout_type
        self.readout_config = readout_config or {}
        
        self.enable_online_learning = enable_online_learning
        self.enable_hierarchical_output = enable_hierarchical_output
        
        # Initialize readout models
        self.custom_readout = None
        self.online_learner = None
        self.hierarchical_output = None
        
        if readout_type == 'mlp':
            self._initialize_mlp_readout()
        elif readout_type == 'online':
            self._initialize_online_readout()
        elif readout_type == 'hierarchical' or enable_hierarchical_output:
            self._initialize_hierarchical_readout(hierarchical_configs)
        
        # State scaling factors based on ESN parameters
        self._state_scaling_factor = 1.0
        self._update_state_scaling()
    
    def reset_state(self) -> None:
        """Reset the internal reservoir state."""
        super().reset_state()  # Call parent method
        # Reset ESN-specific state
        if hasattr(self, '_prev_activations'):
            delattr(self, '_prev_activations')
    
    def _apply_persona_configuration(self):
        """Apply chatbot persona configuration by adjusting ESN parameters."""
        if self.persona_type in ['conservative', 'balanced', 'creative']:
            # Use predefined personality mappings
            for param_name, param_info in self.esn_params.items():
                if 'personality_mapping' in param_info:
                    if self.persona_type in param_info['personality_mapping']:
                        setattr(self, param_name, param_info['personality_mapping'][self.persona_type])
        
        print(f"Applied persona '{self.persona_type}' configuration:")
        print(f"  Spectral radius: {self.spectral_radius} (stability/creativity)")
        print(f"  Leaking rate: {self.leaking_rate} (memory persistence)")
        print(f"  Input scaling: {self.input_scaling} (input sensitivity)")
        print(f"  Density: {self.density} (feature complexity)")
    
    def _update_state_scaling(self):
        """Update state scaling factor based on ESN parameters."""
        # Combine multiple ESN parameters into a unified scaling factor
        self._state_scaling_factor = (
            self.spectral_radius * 
            self.input_scaling * 
            (1.0 + self.bias_scaling)
        )
    
    def _initialize_mlp_readout(self):
        """Initialize multi-layer perceptron readout."""
        config = self.readout_config
        self.custom_readout = MultiLayerReadout(
            input_size=self.units,
            output_size=config.get('output_size', 1),
            hidden_layers=config.get('hidden_layers', [256, 128]),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.1),
            use_torch=config.get('use_torch', True)
        )
    
    def _initialize_online_readout(self):
        """Initialize online learning readout."""
        config = self.readout_config
        self.online_learner = OnlineLearner(
            input_size=self.units,
            output_size=config.get('output_size', 1),
            learning_rate=config.get('learning_rate', 0.01),
            forgetting_factor=config.get('forgetting_factor', 0.99)
        )
    
    def _initialize_hierarchical_readout(self, hierarchical_configs: Optional[List[Dict[str, Any]]]):
        """Initialize hierarchical output system."""
        if hierarchical_configs is None:
            # Default hierarchical configuration
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
                    'readout_type': 'mlp',
                    'readout_params': {
                        'hidden_layers': [128, 64],
                        'activation': 'relu',
                        'use_torch': False
                    }
                }
            ]
        
        self.hierarchical_output = HierarchicalOutput(
            input_size=self.units,
            output_configs=hierarchical_configs
        )
    
    def _apply_esn_transformations(self, activations: np.ndarray) -> np.ndarray:
        """
        Apply ESN parameter transformations to RWKV activations.
        
        This maps ReservoirPy ESN concepts to RWKV hidden state processing.
        """
        # Apply spectral radius scaling (affects dynamics stability)
        activations = activations * self.spectral_radius
        
        # Apply leaking rate (temporal decay simulation)
        if hasattr(self, '_prev_activations') and self.leaking_rate < 1.0:
            # Ensure shapes match for broadcasting
            if self._prev_activations.shape != activations.shape:
                # Reset previous activations if shape mismatch
                self._prev_activations = np.zeros_like(activations)
            
            activations = (
                self.leaking_rate * activations + 
                (1 - self.leaking_rate) * self._prev_activations
            )
        self._prev_activations = activations.copy()
        
        # Apply input scaling effects (sensitivity)
        activations = activations * self.input_scaling
        
        # Apply density effects (feature selection/sparsity)
        if self.density < 1.0:
            # Randomly mask features to simulate sparse connectivity
            mask = np.random.random(activations.shape) < self.density
            activations = activations * mask
        
        # Apply bias scaling
        if self.bias_scaling > 0:
            bias = np.ones_like(activations) * self.bias_scaling
            activations = activations + bias
        
        # Apply noise for creativity/variability
        if self.noise_scaling > 0:
            noise = np.random.normal(0, self.noise_scaling, activations.shape)
            activations = activations + noise
        
        return activations
    
    def _get_reservoir_activations(
        self, 
        input_sequence: Union[List[int], np.ndarray],
        return_states: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Enhanced reservoir activation extraction with ESN transformations.
        """
        # Get base RWKV activations
        activations = super()._get_reservoir_activations(input_sequence, return_states)
        
        if return_states:
            activations, states = activations
        
        # Apply ESN parameter transformations
        transformed_activations = self._apply_esn_transformations(activations)
        
        if return_states:
            return transformed_activations, states
        return transformed_activations
    
    def fit(
        self, 
        X: Union[List, np.ndarray], 
        y: Union[np.ndarray, Dict[str, np.ndarray]], 
        warmup: int = 0,
        hierarchical_targets: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Enhanced training with support for various readout types.
        
        Parameters
        ----------
        X : array-like
            Input sequences
        y : array-like or dict
            Target outputs. Can be dict for hierarchical outputs.
        warmup : int
            Number of warmup steps
        hierarchical_targets : dict, optional
            Targets for hierarchical readouts
        """
        if self.readout_type == 'ridge':
            # Use parent class Ridge regression
            super().fit(X, y, warmup)
        
        elif self.readout_type == 'mlp':
            # Train MLP readout
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], (list, np.ndarray)):
                all_activations = []
                all_targets = []
                
                for seq, target in zip(X, y):
                    activations = self._get_reservoir_activations(seq)
                    if warmup > 0:
                        activations = activations[warmup:]
                        target = target[warmup:] if target.ndim > 1 else target
                    
                    all_activations.append(activations)
                    if target.ndim == 1:
                        target = np.repeat(target.reshape(1, -1), len(activations), axis=0)
                    all_targets.append(target)
                
                X_train = np.concatenate(all_activations)
                y_train = np.vstack(all_targets)
            else:
                X_train = self._get_reservoir_activations(X)
                y_train = y
                if warmup > 0:
                    X_train = X_train[warmup:]
                    y_train = y_train[warmup:]
            
            self.custom_readout.fit(X_train, y_train)
            self._is_trained = True
        
        elif self.readout_type == 'online':
            # Train online learner incrementally
            if isinstance(X, list):
                for seq, target in zip(X, y):
                    activations = self._get_reservoir_activations(seq)
                    if warmup > 0:
                        activations = activations[warmup:]
                        target = target[warmup:] if target.ndim > 1 else target
                    
                    for i in range(len(activations)):
                        self.online_learner.update(
                            activations[i:i+1], 
                            target[i:i+1] if target.ndim > 1 else target.reshape(1, -1)
                        )
            else:
                activations = self._get_reservoir_activations(X)
                if warmup > 0:
                    activations = activations[warmup:]
                    y = y[warmup:]
                
                for i in range(len(activations)):
                    self.online_learner.update(activations[i:i+1], y[i:i+1])
            
            self._is_trained = True
        
        elif self.readout_type == 'hierarchical':
            # Train hierarchical outputs
            if isinstance(X, list):
                all_activations = []
                for seq in X:
                    activations = self._get_reservoir_activations(seq)
                    all_activations.append(activations)
                X_train = np.concatenate(all_activations)
            else:
                X_train = self._get_reservoir_activations(X)
            
            # Prepare hierarchical targets
            if hierarchical_targets is None and isinstance(y, dict):
                hierarchical_targets = y
            elif hierarchical_targets is None:
                # Create default hierarchical targets
                hierarchical_targets = {
                    'readout_0_1': y,
                    'readout_1_5': y[::5] if len(y) > 5 else y,
                    'readout_2_10': y[::10] if len(y) > 10 else y
                }
            
            self.hierarchical_output.fit(X_train, hierarchical_targets)
            self._is_trained = True
    
    def predict(
        self, 
        X: Union[List[int], np.ndarray], 
        reset_state: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Enhanced prediction with support for various readout types.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if reset_state:
            self.reset_state()
        
        activations = self._get_reservoir_activations(X)
        
        if self.readout_type == 'ridge':
            return super().predict(X, reset_state=False)  # Already got activations
        
        elif self.readout_type == 'mlp':
            return self.custom_readout.predict(activations)
        
        elif self.readout_type == 'online':
            return self.online_learner.predict(activations)
        
        elif self.readout_type == 'hierarchical':
            return self.hierarchical_output.predict(activations)
        
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")
    
    def update_online(self, x: np.ndarray, y: np.ndarray):
        """
        Update model weights online (for online learning readouts).
        """
        if self.readout_type != 'online':
            raise ValueError("Online updates only available with online readout type")
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")
        
        activations = self._get_reservoir_activations(x)
        # For online updates, use the last activation (most recent state)
        if activations.ndim > 1 and activations.shape[0] > 1:
            activations = activations[-1:, :]  # Use last time step
        elif activations.ndim == 1:
            activations = activations.reshape(1, -1)
        
        self.online_learner.update(activations, y)
    
    def get_esn_parameter_info(self) -> Dict[str, Any]:
        """
        Get detailed information about ESN parameter mappings.
        """
        current_values = {
            'spectral_radius': self.spectral_radius,
            'leaking_rate': self.leaking_rate,
            'input_scaling': self.input_scaling,
            'density': self.density,
            'bias_scaling': self.bias_scaling,
            'noise_scaling': self.noise_scaling
        }
        
        info = {
            'current_values': current_values,
            'persona_type': self.persona_type,
            'parameter_mappings': self.esn_params,
            'state_scaling_factor': self._state_scaling_factor
        }
        
        return info
    
    def set_persona(self, persona_type: str):
        """
        Dynamically change chatbot persona by adjusting ESN parameters.
        """
        self.persona_type = persona_type
        self._apply_persona_configuration()
        self._update_state_scaling()
        print(f"Persona changed to '{persona_type}'")
    
    def batch_predict(
        self, 
        X_batch: List[Union[List[int], np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Efficient batch prediction for multiple sequences.
        """
        predictions = []
        for X in X_batch:
            # Reset state for each sequence to avoid interference
            self.reset_state()
            pred = self.predict(X, reset_state=False)  # Don't reset again
            predictions.append(pred)
        return predictions
    
    def compare_with_traditional_esn(
        self, 
        X: Union[List, np.ndarray], 
        y: np.ndarray,
        warmup: int = 0
    ) -> Dict[str, float]:
        """
        Compare performance with traditional ESN (if ReservoirPy is available).
        """
        if not HAS_RESERVOIRPY:
            return {"error": "ReservoirPy not available for comparison"}
        
        # Train our model
        self.fit(X, y, warmup=warmup)
        our_score = self.score(X, y, warmup=warmup)
        
        # Create traditional ESN
        try:
            import reservoirpy as rpy
            from reservoirpy.nodes import Reservoir, Ridge
            
            # Convert input to proper format for ReservoirPy
            if isinstance(X, list):
                X_rpy = np.array([np.array(seq) for seq in X])
            else:
                X_rpy = X
            
            reservoir = Reservoir(
                units=self.units,
                lr=self.leaking_rate,
                sr=self.spectral_radius,
                input_scaling=self.input_scaling,
                rc_connectivity=self.density
            )
            readout = Ridge(ridge=1e-6)
            
            esn = reservoir >> readout
            esn.fit(X_rpy, y)
            traditional_score = esn.score(X_rpy, y)
            
            return {
                "rwkv_esn_score": our_score,
                "traditional_esn_score": traditional_score,
                "improvement": our_score - traditional_score
            }
        
        except Exception as e:
            return {"error": f"Traditional ESN comparison failed: {str(e)}"}


# Convenience function for creating enhanced reservoir with persona
def create_chatbot_reservoir(
    shared_library: RWKVSharedLibrary,
    model_path: str,
    persona_type: str = 'balanced',
    advanced_features: bool = True,
    **kwargs
) -> EnhancedReservoirRWKV:
    """
    Create an enhanced reservoir computing system optimized for chatbot personalities.
    
    Parameters
    ----------
    shared_library : RWKVSharedLibrary
        rwkv.cpp shared library
    model_path : str
        Path to RWKV model
    persona_type : str
        Chatbot personality type ('conservative', 'balanced', 'creative', etc.)
    advanced_features : bool
        Enable advanced features (hierarchical outputs, online learning)
    
    Returns
    -------
    EnhancedReservoirRWKV
        Configured reservoir computing system
    """
    config = {
        'persona_type': persona_type,
        'readout_type': 'hierarchical' if advanced_features else 'ridge',
        'enable_online_learning': advanced_features,
        'enable_hierarchical_output': advanced_features
    }
    config.update(kwargs)
    
    return EnhancedReservoirRWKV(
        shared_library=shared_library,
        model_path=model_path,
        **config
    )