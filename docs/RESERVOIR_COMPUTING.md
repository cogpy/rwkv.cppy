# Reservoir Computing with RWKV.cpp

This document describes the reservoir computing implementation using RWKV as the reservoir layer, providing a ReservoirPy-compatible API.

## Overview

Reservoir Computing is a framework for recurrent neural networks where a large "reservoir" of neurons with fixed, randomly initialized weights serves as a dynamic memory. Only the readout layer (output weights) is trained, making the approach computationally efficient.

In this implementation:
- **RWKV model serves as the reservoir** with fixed weights
- **Hidden states** from RWKV are used as reservoir activations  
- **Ridge regression** is used for the trainable readout layer
- **ReservoirPy-compatible API** is provided for ease of use

## Key Concepts

### Traditional Reservoir Computing
```
Input → Reservoir (fixed weights) → Readout (trainable) → Output
```

### RWKV-based Reservoir Computing
```
Token Sequence → RWKV Model (fixed) → Ridge Regression (trainable) → Output
```

## API Reference

### ReservoirRWKV Class

The main class providing reservoir computing functionality.

#### Constructor

```python
from rwkv_cpp import RWKVSharedLibrary, ReservoirRWKV

library = RWKVSharedLibrary("path/to/librwkv.so")
reservoir = ReservoirRWKV(
    shared_library=library,
    model_path="path/to/model.bin",
    units=512,                    # Number of reservoir units (≤ n_embed)
    alpha=1e-6,                  # Ridge regression regularization
    ridge_solver='auto',         # Ridge regression solver
    use_bias=True,              # Use bias in readout layer
    thread_count=None,          # RWKV thread count
    gpu_layer_count=0,          # GPU layers for RWKV
    use_numpy=True,             # Use numpy (vs PyTorch)
    dtype=np.float32            # Data type
)
```

#### Parameters

- `shared_library`: RWKVSharedLibrary instance
- `model_path`: Path to RWKV model in ggml format
- `units`: Number of reservoir units to use (default: full embedding size)
- `alpha`: Ridge regression regularization parameter (default: 1e-6)
- `ridge_solver`: Solver for Ridge regression ('auto', 'svd', 'cholesky', etc.)
- `use_bias`: Whether to include bias in readout layer (default: True)
- `thread_count`: Thread count for RWKV (default: CPU count / 2)
- `gpu_layer_count`: Number of layers to offload to GPU (default: 0)
- `use_numpy`: Use numpy arrays vs PyTorch tensors (default: True)
- `dtype`: Data type for computations (default: np.float32)

#### Methods

##### `fit(X, y, warmup=0)`

Train the readout layer on the given data.

```python
# Single sequence
X = [1, 2, 3, 4, 5]  # Token sequence
y = np.array([[0.5]])  # Target output

reservoir.fit(X, y)

# Multiple sequences
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y = np.array([[0.1], [0.5], [0.9]])

reservoir.fit(X, y, warmup=2)  # Skip first 2 time steps
```

##### `predict(X, reset_state=True)`

Predict outputs for input sequence(s).

```python
predictions = reservoir.predict([1, 2, 3, 4, 5])
print(predictions.shape)  # (seq_len, n_outputs)
```

##### `run(X, reset_state=True)`

Get raw reservoir activations without applying readout layer.

```python
activations = reservoir.run([1, 2, 3, 4, 5])
print(activations.shape)  # (seq_len, units)
```

##### `score(X, y, warmup=0)`

Evaluate model performance (R² score).

```python
r2_score = reservoir.score(X_test, y_test, warmup=2)
print(f"R² score: {r2_score:.4f}")
```

##### `reset_state()`

Reset internal reservoir state.

```python
reservoir.reset_state()
```

#### Properties

- `n_vocab`: Vocabulary size of underlying RWKV model
- `n_embed`: Embedding dimension of RWKV model  
- `n_layer`: Number of layers in RWKV model
- `units`: Number of reservoir units used
- `is_trained`: Whether readout layer has been trained

## Usage Examples

### Time Series Prediction

```python
import numpy as np
from rwkv_cpp import RWKVSharedLibrary, ReservoirRWKV

# Initialize
library = RWKVSharedLibrary("librwkv.so")
reservoir = ReservoirRWKV(
    shared_library=library,
    model_path="model.bin",
    units=256,
    alpha=1e-4
)

# Generate sine wave sequences as token sequences
def generate_sine_data(n_seq=100, seq_len=50):
    sequences = []
    targets = []
    
    for i in range(n_seq):
        # Create sine wave
        t = np.linspace(0, 4*np.pi, seq_len+1)
        sine = np.sin(t + np.random.random() * 2 * np.pi)
        
        # Discretize to tokens [0, 255]
        tokens = ((sine + 1) / 2 * 255).astype(int)
        
        sequences.append(tokens[:-1].tolist())
        targets.append([sine[-1]])  # Predict next value
    
    return sequences, np.array(targets)

# Generate data
X_train, y_train = generate_sine_data(80)
X_test, y_test = generate_sine_data(20)

# Train
reservoir.fit(X_train, y_train, warmup=10)

# Evaluate
test_score = reservoir.score(X_test, y_test, warmup=10)
print(f"Test R² score: {test_score:.4f}")

# Predict
predictions = reservoir.predict(X_test[0])
```

### Memory Task

```python
# Generate memory task: predict sum of first k tokens
def generate_memory_task(n_seq=100, seq_len=30, k=3):
    sequences = []
    targets = []
    
    for _ in range(n_seq):
        seq = np.random.randint(0, 10, seq_len).tolist()
        target = sum(seq[:k]) / (10 * k)  # Normalized sum
        
        sequences.append(seq)
        targets.append([target])
    
    return sequences, np.array(targets)

# Generate data
X_train, y_train = generate_memory_task(100, 30, 3)
X_test, y_test = generate_memory_task(20, 30, 3)

# Train and evaluate
reservoir.fit(X_train, y_train, warmup=5)
score = reservoir.score(X_test, y_test, warmup=5)
print(f"Memory task R² score: {score:.4f}")
```

### Classification Task

```python
# Binary classification based on sequence properties
def generate_classification_data(n_seq=200, seq_len=20):
    sequences = []
    targets = []
    
    for _ in range(n_seq):
        seq = np.random.randint(0, 100, seq_len).tolist()
        # Classify: even/odd sum
        label = (sum(seq) % 2)
        
        sequences.append(seq)
        targets.append([float(label)])
    
    return sequences, np.array(targets)

X_train, y_train = generate_classification_data(150)
X_test, y_test = generate_classification_data(50)

# Train
reservoir.fit(X_train, y_train, warmup=3)

# Evaluate
predictions = reservoir.predict(X_test[0])
predicted_class = int(predictions[-1, 0] > 0.5)
true_class = int(y_test[0, 0])
print(f"Predicted: {predicted_class}, True: {true_class}")
```

## Performance Considerations

### Model Selection
- **Smaller models** (e.g., 169M parameters) are often sufficient for reservoir tasks
- **Larger models** may provide richer representations but require more memory
- **Quantized models** (Q4_0, Q5_1) can reduce memory usage with minimal quality loss

### Hyperparameter Tuning
- `units`: Start with 128-512 units, increase if needed
- `alpha`: Ridge regularization, try values from 1e-8 to 1e-2
- `warmup`: Skip initial time steps to allow reservoir to stabilize (typically 5-20)

### Memory Usage
- Each reservoir unit stores one float32 value per time step
- Memory scales as: `seq_len × units × batch_size × 4 bytes`
- Use smaller `units` for longer sequences or larger batches

### Computational Efficiency
- RWKV evaluation is efficient with O(n) complexity (vs O(n²) for transformers)
- Ridge regression training is fast for moderate output dimensions
- GPU acceleration available for RWKV layers

## Comparison with Traditional ESNs

| Aspect | Traditional ESN | RWKV-based |
|--------|----------------|------------|
| Reservoir weights | Random initialization | Pre-trained RWKV |
| Memory capacity | Limited by reservoir size | Enhanced by RWKV's sequential processing |
| Interpretability | Random features | Learned language representations |
| Training data | Task-specific only | Benefits from pre-training |
| Scalability | Limited by matrix operations | Efficient sequential processing |

## Integration with ReservoirPy

This implementation provides a ReservoirPy-compatible API, allowing easy migration:

```python
# ReservoirPy style
from reservoirpy.nodes import Reservoir, Ridge

# RWKV style  
from rwkv_cpp import ReservoirRWKV

# Similar API patterns
reservoir = ReservoirRWKV(...)
reservoir.fit(X, y)
predictions = reservoir.predict(X)
```

## Requirements

- Python 3.7+
- NumPy
- scikit-learn (for Ridge regression)
- Optional: PyTorch (for tensor operations)
- Built rwkv.cpp library

## Limitations

- Input must be token sequences (integers)
- RWKV model weights are fixed (not fine-tuned)
- Ridge regression readout (no deep readout layers)
- Single readout layer (no hierarchical outputs)

## Future Extensions

Potential improvements and extensions:

1. **Multi-layer readouts**: Deep neural network readout layers
2. **Online learning**: Incremental training of readout weights  
3. **Hierarchical outputs**: Multiple readout layers at different time scales
4. **Custom readouts**: Support for other regression/classification methods
5. **Model fine-tuning**: Optional fine-tuning of RWKV weights
6. **Batch processing**: Efficient batch evaluation of multiple sequences