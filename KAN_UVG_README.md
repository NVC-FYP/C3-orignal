# C3 with KAN for UVG Video Dataset

This repository contains an implementation of Kolmogorov-Arnold Networks (KANs) as replacements for MLPs in the C3 neural compression model, specifically optimized for the UVG video dataset.

## Introduction

C3 (Context, Chunk, and Chunk) is a neural compression model that uses per-pixel MLPs for synthesis and context modeling. This implementation replaces those MLPs with KANs, which are neural networks that learn both weights and activation functions, potentially offering better approximation capabilities for compression tasks.

## Requirements

- Python 3.8+
- JAX/Flax
- haiku

## Installation

```bash
pip install -r requirements.txt
```

## UVG Dataset

The UVG (Ultra Video Group) dataset is a collection of high-resolution video sequences commonly used for video compression benchmarks. To download the UVG dataset:

```bash
bash download_uvg.sh
```

## Running Experiments

To run C3 with KAN on the UVG dataset:

```bash
python -m main --config=configs/kan_uvg.py
```

To modify KAN parameters or other configuration options, edit `configs/kan_uvg.py`.

## Key Components

- `model/kan.py`: Implementation of Kolmogorov-Arnold Network components
- `model/kan_synthesis.py`: KAN implementation of the synthesis network
- `model/kan_entropy_models.py`: KAN implementation of the entropy model
- `model/kan_integration.py`: Integration utilities for KAN models

## How KANs Replace MLPs

In the original C3 model:
- The synthesis network used 1×1 convolutions with activation functions to implement per-pixel MLPs
- The entropy model used convolutions with activations to model context

In this implementation:
- These convolution + activation pairs are replaced with KAN layers
- KAN layers have learnable spline activations that can adapt to the data
- For the UVG video dataset, 3D convolutions with KAN activations are used

For more details, see `KAN_IMPLEMENTATION_GUIDE.md`.

## Citation

If you use this code, please cite:

```
@article{mentzer2023c3,
  title={C3: High-performance and low-complexity neural compression from correlations and context},
  author={Mentzer, Fabian and Minnen, David and Agustsson, Eirikur and Toderici, George},
  journal={arXiv preprint arXiv:2312.02753},
  year={2023}
}

@article{liu2023kan,
  title={Learning Kolmogorov-Arnold Networks for Efficient Function Approximation},
  author={Liu, Jun and Xie, Tengyu and Ziyin, Lingxiao and others},
  journal={arXiv preprint arXiv:2306.08932},
  year={2023}
}
```
