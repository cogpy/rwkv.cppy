# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rwkv.cpp is a C/C++ port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) built on [ggml](https://github.com/ggerganov/ggml). It provides efficient CPU-focused inference for RWKV language models with support for FP32, FP16, and quantized INT4/INT5/INT8 formats. Optional GPU acceleration is available via cuBLAS/hipBLAS.

The project includes a Python wrapper and **ReservoirRWKV** - a Reservoir Computing implementation using RWKV as the reservoir layer with a ReservoirPy-compatible API.

## Project Structure

```
rwkv.cppy/
├── rwkv.cpp / rwkv.h          # Core C library (public API in rwkv.h)
├── rwkv_*.inc                  # C implementation modules (graph, eval, quantize, etc.)
├── ggml/                       # ggml submodule for tensor operations
├── python/
│   ├── rwkv_cpp/              # Python package
│   │   ├── rwkv_cpp_model.py  # High-level model wrapper
│   │   ├── rwkv_cpp_shared_library.py  # Library bindings
│   │   ├── reservoir.py       # ReservoirRWKV implementation
│   │   ├── enhanced_reservoir.py  # Advanced ESN features
│   │   └── rwkv_world_tokenizer.py  # Tokenizer for World models
│   ├── convert_pytorch_to_ggml.py  # Model conversion
│   ├── quantize.py            # Model quantization
│   ├── generate_completions.py  # Text generation
│   ├── chat_with_bot.py       # Chat interface
│   └── *_example.py           # Usage examples
├── tests/                     # C test suite with pre-built test models
├── docs/                      # Documentation
└── extras/                    # Additional utilities
```

## Build Commands

Build the native library with CMake:

```bash
cmake .
cmake --build . --config Release
```

This produces `librwkv.so` (Linux), `librwkv.dylib` (macOS), or `bin\Release\rwkv.dll` (Windows).

For GPU support:
```bash
cmake . -DRWKV_CUBLAS=ON  # NVIDIA
cmake . -DRWKV_HIPBLAS=ON  # AMD
cmake --build . --config Release
```

## Python Usage

### Requirements
- Python 3.7+
- numpy
- scikit-learn (for reservoir features)
- tokenizers (for Pile/Raven models)
- Optional: PyTorch

### Converting Models
```bash
python python/convert_pytorch_to_ggml.py <input.pth> <output.bin> FP16
python python/quantize.py <input.bin> <output.bin> Q5_1
```

### Running Tests
```bash
# Python tests
python python/test_reservoir.py
python python/test_enhanced_reservoir.py

# C tests (requires built library and test models in tests/)
ctest --test-dir .
```

## Code Style Guidelines

### C/C++
- 4 spaces for indentation
- Use 1TBS (One True Brace Style) - braces on same line
- Prefix top-level functions/structs with `rwkv_`
- Mark non-API functions as `static`
- Mark immutable arguments as `const`
- Max line length: 180 characters

### Python
- 2 spaces for indentation
- Type hints required for functions and parameters
- Use `typing` module types (`List`, `Dict`) instead of built-ins
- Specify `-> None` for void functions

## Key Architecture Notes

### RWKV Model
- Supports versions v4, v5, v6, and v7
- O(n) complexity vs O(n²) for transformers - CPU-friendly for long contexts
- State-based: only requires previous step's state for inference

### Reservoir Computing
The `ReservoirRWKV` class uses RWKV as a fixed-weight reservoir:
- Input tokens → RWKV (frozen) → Hidden states → Ridge regression (trainable) → Output
- Provides ReservoirPy-compatible API: `fit()`, `predict()`, `run()`, `score()`

### Enhanced Reservoir (`EnhancedReservoirRWKV`)
Additional features:
- `ESNParameterMapping`: Maps ESN parameters to RWKV concepts
- `MultiLayerReadout`: Deep MLP readout layers
- `OnlineLearner`: Incremental weight updates
- `HierarchicalOutput`: Multi-timescale outputs

## Test Models

Pre-built tiny test models are available in `tests/`:
- `tiny-rwkv-4v0-660K-*.bin` (RWKV v4)
- `tiny-rwkv-5v1-730K-*.bin` (RWKV v5.1)
- `tiny-rwkv-5v2-730K-*.bin` (RWKV v5.2)
- `tiny-rwkv-6v0-3m-*.bin` (RWKV v6)
- `tiny-rwkv-7v0-834K-*.bin` (RWKV v7)

Available in formats: FP32, FP16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

## Common Development Tasks

### Adding a New Quantization Format
1. Update `rwkv_quantize.inc` with the new format
2. Add format to the switch statements in `rwkv_file_format.inc`
3. Update `python/quantize.py` to support the new format

### Extending Reservoir Computing
1. New features go in `python/rwkv_cpp/enhanced_reservoir.py`
2. Update `python/rwkv_cpp/__init__.py` exports
3. Add examples to `python/advanced_reservoir_example.py`
4. Update `docs/RESERVOIR_COMPUTING.md`

### Testing Changes
- C tests are in `tests/test_*.c`
- Python tests in `python/test_*.py`
- Run `python python/measure_pexplexity.py` for quality validation
