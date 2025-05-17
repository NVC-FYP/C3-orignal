# Replacing MLPs with Kolmogorov-Arnold Networks (KANs) in C3 Neural Compression

This guide explains how to replace Multi-Layer Perceptrons (MLPs) with Kolmogorov-Arnold Networks (KANs) in the C3 neural compression model, specifically for the UVG video dataset.

## Overview

The C3 neural compression model uses MLPs in two main components:

1. **Synthesis Network (`synthesis.py`)**: An elementwise MLP implemented as 1×1 convolutions that maps latent representations to RGB outputs.
2. **Entropy/Context Model (`entropy_models.py`)**: MLPs mapping local context to Laplace distribution parameters.

For the UVG (video) dataset, C3 extends these networks to 3D using masked 3D convolutions.

## Files Created

We've created the following files to implement KAN replacements for MLPs:

1. **`kan_synthesis.py`**: Contains `KANSynthesis` class that replaces MLPs with KANs in the synthesis network
2. **`kan_entropy_models.py`**: Contains `KANAutoregressiveEntropyModelConvVideo` class that replaces MLPs with KANs in the entropy model
3. **`kan_integration.py`**: Utility functions to integrate KAN models into the C3 architecture
4. **`configs/kan_uvg.py`**: Configuration file for KAN-based UVG experiments

## How to Use

### 1. Configuration

To enable KAN for your experiments, use the KAN-specific configuration:

```bash
python -m main --config=configs/kan_uvg.py
```

The KAN configuration modifies the base UVG configuration by:
- Setting `use_kan=True` to enable KAN models
- Adding KAN-specific parameters (`num_knots=15`, `spline_range=3.0`)

You can also customize these parameters:

```python
# In configs/kan_uvg.py
exp.model.kan.num_knots = 20  # Increase number of knots
exp.model.kan.spline_range = 4.0  # Change spline range
```

### 2. Key Changes Made

#### Synthesis Network (MLPs → KANs)

The original synthesis network used 1×1 convolutions followed by activation functions (e.g., GELU) to implement pixel-wise MLPs. We've replaced these with KAN-based convolutional layers (`KANConv2D` and `KANConv3D`) that have built-in spline activations.

#### Entropy Model (MLPs → KANs)

The entropy model used MLPs implemented as convolutions to predict Laplace distribution parameters. We've replaced these with KAN layers that use learnable spline activations.

## KAN Architecture

The Kolmogorov-Arnold Networks implemented in this codebase:

1. Use piecewise linear splines as activation functions
2. Learn both the weights and the activation functions during training
3. Support 2D (images) and 3D (video) convolutions

The key component is the `SplineActivation` class, which implements a learnable piecewise linear function:

```python
class SplineActivation(hk.Module):
    """Learnable spline activation function for KAN."""
    
    def __init__(
        self,
        num_knots: int = 10,
        spline_range: float = 3.0,
        init_scheme: str = "relu_like",
    ):
        # ...
```

## Important Parameters

- **`num_knots`**: The number of knots in the spline activation (higher = more expressive)
- **`spline_range`**: The input range for the spline activation (similar to clipping values)

## Performance Considerations

- KANs may require more parameters than MLPs due to the additional learnable activation functions
- Training may be slightly slower, but inference speed should be comparable
- You may want to adjust the learning rate or training schedule for optimal performance

## Extending to Other Datasets

To use KANs with other datasets:

1. Create a new configuration file (e.g., `configs/kan_kodak.py`)
2. Set `use_kan=True` and configure KAN parameters
3. Use the same integration utilities for model creation

The KAN implementation is dataset-agnostic and should work with any dataset supported by C3.

## References

- C3 Neural Compression: [Paper](https://arxiv.org/abs/2312.02753)
- Kolmogorov-Arnold Networks: [Paper](https://arxiv.org/abs/2306.08932)
