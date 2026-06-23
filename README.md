**Deep Learning Enhanced Ensemble Kalman Filter for Sea Surface Temperature Forecasting**

---

## Overview

Deepcov-EnKF is a novel data assimilation framework that combines deep learning-based sea surface temperature (SST) forecasting with intelligent background error covariance (B) modeling for improved ensemble Kalman filter performance.

### Key Features

- **OceanForecast**: Transformer-based SST forecasting model with periodic boundary handling
- **DL-BEC**: Deep learning module for computing background error covariance matrices
- **End-to-End**: Integrated pipeline for data assimilation and prediction
- **Scalable**: Support for multi-GPU training and distributed computing

### Project Structure

```
Deepcov-EnKF/
├── README.md              # This file
├── forecast/              # SST forecasting module
│   ├── README.md          # Forecast-specific documentation
│   ├── src/               # Source code
│   ├── configs/           # Hydra configuration files
│   └── scripts/           # Training scripts
└── DL-BEC/                # Background error covariance module
    ├── README.md          # DL-BEC specific documentation
    ├── main.py            # Entry point
    ├── train.py           # Training logic
    └── h5dataset.py       # Data generation
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Libx2000/Deepcov-EnKF.git
cd Deepcov-EnKF

# Install forecast dependencies
cd forecast
pip install -r requirements.txt

# Install DL-BEC dependencies
cd ../DL-BEC
pip install -r requirements.txt
```

## Modules

| Module | Description | Location |
|--------|-------------|----------|
| **OceanForecast** | Swin-Transformer based SST predictor | `forecast/` |
| **DL-BEC** | Deep learning covariance matrix estimator | `DL-BEC/` |

## License

This project is licensed under the MIT License.
