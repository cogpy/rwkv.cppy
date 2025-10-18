"""
ReservoirPy-compatible implementation using RWKV as the reservoir layer.

This module provides an Echo State Network implementation where the RWKV model
serves as the reservoir (with fixed weights) and a trainable readout layer
is added on top for different tasks.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any
from .rwkv_cpp_model import RWKVModel
from .rwkv_cpp_shared_library import RWKVSharedLibrary
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ReservoirRWKV:
    """
    A Reservoir Computing implementation using RWKV as the reservoir layer.
    
    This class provides an Echo State Network (ESN) interface where:
    - RWKV model serves as the reservoir with fixed weights
    - Hidden states from RWKV are used as reservoir activations
    - A trainable readout layer (Ridge regression by default) is trained on top
    
    This is compatible with ReservoirPy's API design while leveraging the
    efficiency of rwkv.cpp for the reservoir computations.
    """
    
    def __init__(
        self,
        shared_library: RWKVSharedLibrary,
        model_path: str,
        units: Optional[int] = None,
        alpha: float = 1e-6,
        ridge_solver: str = 'auto',
        use_bias: bool = True,
        thread_count: Optional[int] = None,
        gpu_layer_count: int = 0,
        use_numpy: bool = True,
        dtype: Any = np.float32,
        **kwargs
    ):
        """
        Initialize ReservoirRWKV.
        
        Parameters
        ----------
        shared_library : RWKVSharedLibrary
            rwkv.cpp shared library instance
        model_path : str
            Path to RWKV model file in ggml format
        units : int, optional
            Number of reservoir units to use. If None, uses full RWKV embedding size
        alpha : float, default=1e-6
            Ridge regression regularization parameter
        ridge_solver : str, default='auto'
            Solver for ridge regression ('auto', 'svd', 'cholesky', etc.)
        use_bias : bool, default=True
            Whether to use bias in readout layer
        thread_count : int, optional
            Thread count for RWKV. If None, uses default
        gpu_layer_count : int, default=0
            Number of layers to offload to GPU
        use_numpy : bool, default=True
            Whether to use numpy arrays (vs PyTorch tensors)
        dtype : data type, default=np.float32
            Data type for computations
        """
        
        # Initialize RWKV model
        if thread_count is None:
            import multiprocessing
            thread_count = max(1, multiprocessing.cpu_count() // 2)
            
        self.rwkv_model = RWKVModel(
            shared_library=shared_library,
            model_path=model_path,
            thread_count=thread_count,
            gpu_layer_count=gpu_layer_count
        )
        
        # Model properties
        self.n_vocab = self.rwkv_model.n_vocab
        self.n_embed = self.rwkv_model.n_embed
        self.n_layer = self.rwkv_model.n_layer
        
        # Reservoir configuration
        self.units = units if units is not None else self.n_embed
        if self.units > self.n_embed:
            raise ValueError(f"units ({self.units}) cannot exceed model embedding size ({self.n_embed})")
            
        self.alpha = alpha
        self.ridge_solver = ridge_solver
        self.use_bias = use_bias
        self.use_numpy = use_numpy
        self.dtype = dtype
        
        # Training state
        self._is_trained = False
        self._readout_weights = None
        self._readout_bias = None
        self._ridge_regressor = None
        
        # Reservoir state
        self._reservoir_state = None
        
        # Statistics for input scaling
        self._input_scaling = 1.0
        self._input_shift = 0.0
        
    @property 
    def is_trained(self) -> bool:
        """Check if the readout layer has been trained."""
        return self._is_trained
        
    def reset_state(self) -> None:
        """Reset the internal reservoir state."""
        self._reservoir_state = None
        
    def _process_tokens(self, tokens: Union[List[int], np.ndarray]) -> List[int]:
        """Process input tokens to ensure correct format."""
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return [int(token) for token in tokens]
        
    def _get_reservoir_activations(
        self, 
        input_sequence: Union[List[int], np.ndarray],
        return_states: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get reservoir activations for an input sequence.
        
        Parameters
        ----------
        input_sequence : array-like
            Input token sequence
        return_states : bool, default=False
            Whether to return intermediate states as well
            
        Returns
        -------
        activations : np.ndarray
            Reservoir activations of shape (seq_len, units)
        states : np.ndarray, optional
            Full RWKV states if return_states=True
        """
        tokens = self._process_tokens(input_sequence)
        seq_len = len(tokens)
        
        # Initialize output arrays
        activations = np.zeros((seq_len, self.units), dtype=self.dtype)
        
        if return_states:
            state_size = self.rwkv_model._state_buffer_element_count
            states = np.zeros((seq_len, state_size), dtype=self.dtype)
        
        # Process sequence token by token to get activations
        current_state = self._reservoir_state
        
        for i, token in enumerate(tokens):
            # Get RWKV logits and state
            logits, current_state = self.rwkv_model.eval(
                token=token,
                state_in=current_state,
                use_numpy=self.use_numpy
            )
            
            # Extract reservoir activations (first 'units' elements of state)
            # The state contains various internal states; we use embedding portion
            if self.use_numpy:
                activation = current_state[:self.units].astype(self.dtype)
            else:
                activation = current_state[:self.units].cpu().numpy().astype(self.dtype)
                
            activations[i] = activation
            
            if return_states:
                if self.use_numpy:
                    states[i] = current_state.astype(self.dtype)
                else:
                    states[i] = current_state.cpu().numpy().astype(self.dtype)
        
        # Update internal state
        self._reservoir_state = current_state
        
        if return_states:
            return activations, states
        return activations
        
    def fit(
        self, 
        X: Union[List[List[int]], np.ndarray], 
        y: np.ndarray,
        warmup: int = 0
    ) -> 'ReservoirRWKV':
        """
        Train the readout layer on the given data.
        
        Parameters
        ----------
        X : array-like
            Input sequences, either list of token sequences or array
        y : np.ndarray
            Target outputs of shape (n_samples, n_outputs) or (n_samples,)
        warmup : int, default=0
            Number of initial time steps to discard (warmup period)
            
        Returns
        -------
        self : ReservoirRWKV
            Returns self for method chaining
        """
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for training. Install with: pip install scikit-learn")
            
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Collect all reservoir activations
        all_activations = []
        all_targets = []
        
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], (list, np.ndarray)):
            # List of sequences (each element is itself a sequence)
            if len(X) != len(y):
                raise ValueError(f"Number of sequences ({len(X)}) must match number of targets ({len(y)})")
                
            for seq, target in zip(X, y):
                self.reset_state()  # Reset state for each sequence
                activations = self._get_reservoir_activations(seq)
                
                # Apply warmup
                if warmup > 0:
                    activations = activations[warmup:]
                    if len(activations) == 0:
                        continue
                        
                all_activations.append(activations)
                
                # Handle target alignment
                if target.ndim == 1:
                    # Single target per sequence - repeat for all time steps
                    seq_targets = np.repeat(target.reshape(1, -1), len(activations), axis=0)
                    all_targets.append(seq_targets)
                else:
                    # Multiple targets per sequence
                    seq_targets = target[warmup:] if warmup > 0 else target
                    if len(seq_targets) != len(activations):
                        # Use last target for all time steps
                        seq_targets = np.repeat(target[-1].reshape(1, -1), len(activations), axis=0)
                    all_targets.append(seq_targets)
        else:
            # Single sequence
            self.reset_state()
            activations = self._get_reservoir_activations(X)
            
            # Apply warmup
            if warmup > 0:
                activations = activations[warmup:]
                
            # Handle targets - repeat for each time step if needed
            if y.shape[0] == 1:
                # Single target - repeat for all time steps
                y_expanded = np.repeat(y, len(activations), axis=0)
            else:
                # Multiple targets - use as is (apply warmup if needed)
                y_expanded = y[warmup:] if warmup > 0 else y
                
            all_activations = [activations]
            all_targets = [y_expanded]
            
        # Concatenate all data
        X_reservoir = np.vstack(all_activations)
        y_fit = np.vstack(all_targets)
        
        # Train Ridge regression readout
        self._ridge_regressor = Ridge(
            alpha=self.alpha,
            solver=self.ridge_solver,
            fit_intercept=self.use_bias
        )
        
        self._ridge_regressor.fit(X_reservoir, y_fit)
        
        # Store weights for direct access
        self._readout_weights = self._ridge_regressor.coef_
        if self.use_bias:
            self._readout_bias = self._ridge_regressor.intercept_
        else:
            self._readout_bias = None
            
        self._is_trained = True
        
        return self
        
    def predict(
        self, 
        X: Union[List[int], np.ndarray],
        reset_state: bool = True
    ) -> np.ndarray:
        """
        Predict outputs for the given input sequence.
        
        Parameters
        ----------
        X : array-like
            Input token sequence
        reset_state : bool, default=True
            Whether to reset reservoir state before prediction
            
        Returns
        -------
        predictions : np.ndarray
            Predicted outputs of shape (seq_len, n_outputs)
        """
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction. Call fit() first.")
            
        if reset_state:
            self.reset_state()
            
        # Get reservoir activations
        activations = self._get_reservoir_activations(X)
        
        # Apply readout layer
        predictions = self._ridge_regressor.predict(activations)
        
        # Ensure consistent shape (flatten if single output)
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions
        
    def run(
        self, 
        X: Union[List[int], np.ndarray],
        reset_state: bool = True
    ) -> np.ndarray:
        """
        Run the reservoir without the readout layer (get raw activations).
        
        Parameters
        ----------
        X : array-like
            Input token sequence  
        reset_state : bool, default=True
            Whether to reset reservoir state before running
            
        Returns
        -------
        activations : np.ndarray
            Raw reservoir activations of shape (seq_len, units)
        """
        
        if reset_state:
            self.reset_state()
            
        return self._get_reservoir_activations(X)
        
    def score(
        self, 
        X: Union[List[List[int]], np.ndarray], 
        y: np.ndarray,
        warmup: int = 0
    ) -> float:
        """
        Evaluate the model performance on test data.
        
        Parameters
        ----------
        X : array-like
            Input sequences
        y : np.ndarray
            True target values
        warmup : int, default=0
            Number of initial time steps to discard
            
        Returns
        -------
        score : float
            R^2 coefficient of determination
        """
        
        if not self._is_trained:
            raise RuntimeError("Model must be trained before scoring. Call fit() first.")
            
        # Get predictions
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], (list, np.ndarray)):
            all_predictions = []
            all_targets = []
            
            for seq, target in zip(X, y):
                pred = self.predict(seq, reset_state=True)
                if warmup > 0:
                    pred = pred[warmup:]
                    if target.ndim > 1:
                        target = target[warmup:]
                        
                all_predictions.append(pred)
                
                if target.ndim == 1:
                    # Single target per sequence - repeat for all time steps
                    seq_targets = np.repeat(target.reshape(1, -1), len(pred), axis=0)
                    all_targets.append(seq_targets)
                else:
                    all_targets.append(target)
                    
            # Concatenate predictions and targets correctly
            y_pred = np.concatenate(all_predictions)  # Now always 1D after predict fix
            y_true = np.vstack(all_targets)
            if y_true.ndim > 1 and y_true.shape[1] == 1:
                y_true = y_true.flatten()
        else:
            y_pred = self.predict(X, reset_state=True)
            y_true = y
            
            if warmup > 0:
                y_pred = y_pred[warmup:]
                y_true = y_true[warmup:]
                
        # Calculate R^2 score using sklearn's built-in r2 score
        return r2_score(y_true, y_pred)
        
    def __del__(self):
        """Clean up RWKV model resources."""
        if hasattr(self, 'rwkv_model') and hasattr(self.rwkv_model, '_valid') and self.rwkv_model._valid:
            self.rwkv_model.free()