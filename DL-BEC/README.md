# DL-BEC

**Deep Learning Background Error Covariance Model for Ensemble Kalman Filter**

---

## Overview

DL-BEC is a deep learning module for computing background error covariance matrices (B) in ensemble data assimilation. It uses a block-wise low-rank approximation to efficiently estimate large-scale covariance matrices for the Ensemble Kalman Filter (EnKF).

## Installation

```bash
cd DL-BEC
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python main.py --help
```

#### Generate Training Data

```bash
python main.py generate \
    --output_path ./data/train_data.h5 \
    --m 28800 \
    --n 5000 \
    --num_samples 100 \
    --mask_path ./maskP.npy
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output_path` | Output HDF5 file path | `./data/train_data.h5` |
| `--m` | Data dimension | `28800` |
| `--n` | Number of ensemble members | `5000` |
| `--num_samples` | Number of training samples | `100` |
| `--mask_path` | Mask file path for spatial constraints | `./maskP.npy` |

#### Train Model

```bash
python main.py train \
    --data_path ./data/train_data.h5 \
    --m 28800 \
    --batch_size 64 \
    --block_size 512 \
    --rank 10 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir ./models \
    --gpus 0
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` | HDF5 data file path | *required* |
| `--m` | Data dimension | *required* |
| `--batch_size` | Batch size | `64` |
| `--block_size` | Block size for covariance matrix | `512` |
| `--rank` | Rank of low-rank approximation | `10` |
| `--epochs` | Number of training epochs | `50` |
| `--lr` | Learning rate | `1e-4` |
| `--save_dir` | Model save directory | `./models` |
| `--gpus` | GPU IDs (comma-separated) | `0` |
```
DL-BEC/
├── main.py           # Command line entry point
├── train.py          # Training logic and model definition
│   ├── BlockLowRankCov      # Core model class
│   ├── BlockDiagMatrix      # Block diagonal matrix utility
│   ├── CovarianceLoss       # Custom loss function
│   └── H5PerturbationDataset # Dataset class
├── h5dataset.py      # Training data generation
├── maskP.npy         # Spatial mask for data generation
└── requirements.txt  # Dependencies
```

## Mathematical Formulation

### Covariance Matrix Estimation

Given ensemble perturbations `X ∈ R^(n×m)` with `n` members and `m` grid points:

1. **Centering**: `X_c = X - mean(X)`
2. **Empirical Covariance**: `B = X_c^T @ X_c / (n - 1)`
3. **Low-Rank Approximation**: `B ≈ L @ L^T + diag(D)`

### Block Structure

The covariance matrix is decomposed into `K = m / block_size` blocks:

```
B = diag(B_1, B_2, ..., B_K)
where B_i = L_i @ L_i^T + diag(D_i)
```

## License

MIT License
