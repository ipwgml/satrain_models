# Basic UNet Model for SatRain Dataset

This directory contains a basic UNet implementation for training on the SatRain dataset using the lightning module and data module from the  `satrain_models` package.

## Overview

This model uses the UNet architecture from the `satrain_models` package to
perform the estimation tasks on satellite data.

## Files

- `train.py`: Main training script
- `compute.toml`: Compute configuration
- `dataset.toml`: SatRain Dataset configuration
- `requirements.txt`:
- `README.md` - This documentation file

## Requirements

Install the required dependencies using conda:

```bash
conda install --file requirements.txt
```


### Training

Basic training command:

```bash
python train.py
```
## Configuration

The configuration is split into two files for better organization:

### Model Configuration (`dataset.toml`)
Configures the SatRain input data.

### Input Configuration (`compute.toml`) 
Configures compute settings (accelerator, devices, etc.)

## Logging and Monitoring

Training logs are automatically saved for TensorBoard visualization:

```bash
tensorboard --logdir lightning_logs
```

