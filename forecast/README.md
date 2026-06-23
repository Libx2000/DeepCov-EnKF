# OceanForecast

**Transformer-Based Sea Surface Temperature Forecasting Model**

---

## Overview

OceanForecast is a deep learning model for sea surface temperature (SST) forecasting based on the Swin Transformer architecture. 

## Architecture

### Model Components

- **PeriodicConv2d**: Custom convolution layer handling east-west periodic boundaries
- **WindowAttentionV2**: Window-based cosine attention with continuous relative position bias
- **SwinBlock**: Transformer block with shifted window attention
- **Multi-scale Encoder**: Four stages of feature extraction with progressive downsampling

### Input Variables

| Type | Variables |
|------|-----------|
| **Atmospheric** | land_sea_mask, orography, latitude, longitude, z_500, z_850, msl, t2m, u10, v10 |
| **Oceanic** | thetao_0.5 (sea potential temperature at 0.5m depth) |

### Output Variables

- **thetao_0.5**: Sea potential temperature prediction

## Installation

```bash
cd forecast
pip install -r requirements.txt
```

## Training

### Configuration

Training is managed via Hydra configuration files located in `configs/`:

| Config File | Purpose |
|-------------|---------|
| `train.yaml` | Main training configuration |
| `eval.yaml` | Evaluation configuration |
| `model/oceanforecast_ar2.yaml` | AR2 model configuration |
| `model/oceanforecast_ar4.yaml` | AR4 model configuration |
| `model/oceanforecast_pretrain.yaml` | Pre-training configuration |

### Pre-training

```bash
python src/train.py --config-name train.yaml model=oceanforecast_pretrain
```

### Fine-tuning

```bash
# 2-step ahead forecasting
python src/train.py --config-name train.yaml model=oceanforecast_ar2

# 4-step ahead forecasting
python src/train.py --config-name train.yaml model=oceanforecast_ar4
```


## Project Structure

```
forecast/
├── src/
│   ├── datamodules/            # HDF5 data loading
│   │   ├── h5datamodule_oceanforecast.py
│   │   └── h5dataset_oceanforecast.py
│   ├── models/
│   │   └── forecast/
│   │       └── oceanforecast/
│   │           ├── __init__.py
│   │           └── arch.py     # Model architecture
│   ├── tasks/                  # Training/evaluation tasks
│   │   ├── train_task.py
│   │   └── eval_task.py
│   ├── evaluate/               # Evaluation and inference
│   │   ├── medium_forecast/
│   │   └── inference_forecast.py
│   └── utils/                  # Utility functions
│       ├── crps.py             # CRPS metric
│       ├── weighted_acc_rmse.py # Latitude-weighted metrics
│       └── score.py            # Scoring utilities
├── configs/                    # Hydra configurations
├── scripts/                    # Training scripts
└── requirements.txt            # Dependencies
```

## License

MIT License
