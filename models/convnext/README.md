# ConvNeXt + UNet Decoder for SatRain Dataset

This directory contains a ConvNeXt backbone with a UNet-style decoder for precipitation regression on the SatRain dataset.

## Overview

Uses `convnext_tiny` / `small` / `base` / `large` from torchvision as encoder + custom UNet-style decoder with skip connections.  
Designed for full-resolution spatial precipitation output (regression task).

**Main modifications compared to original ConvNeXt:**
- Removed: Classification head, global pooling
- Added: UNet decoder (4 stages), skip connections (3 levels)
- Modified: First conv for flexible input channels
- Changed: Output from (B, 1000) to (B, 1, H, W)
- Inserted: Skip connection adapters (1×1 convs)
- Replaced: CrossEntropyLoss → MSELoss
- Implemented: Progressive spatial upsampling (5 stages)


## Files

- `train.py`          – Training script
- `test.py`           – Evaluation script
- `compute.toml`      – Compute & training settings
- `dataset.toml`      – Dataset & input/output configuration
- `requirements.txt`  – Required packages
- `README.md`         – This file

## Requirements

```bash
pip install -r requirements.txt
## Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```


### Training

The basic training command is:

```bash
python train.py
```

The configuration of the UNet models and the compute environment are read from the ``dataset.toml`` and ``compute.toml`` files. Therefore no additional parameters are required for the train command.

Each training will create several artifacts:
    
- Logs are stored in the ``lightning_logs`` folder. User ``tensorboard --logdir lightning_logs`` to view them.
- Model checkpoints are stored in the ``checkpoints`` folder. 
- The logged training metrics are also stored as NetCDF files in the ``netcdf_metrics`` folder.
- Finally, the model weights are stored in the ``models`` folder.


### Testing

To evaluate a model on the SatRain testing data run

```bash
python test.py models/model_name.pt
```

This should work for both final stored models as well as checkpoint files.

## Configuration

The configuration is split into two files for better organization:

### Model Configuration (`dataset.toml`)

This file contains the configuration of the SatRain dataset. It is used to
configure the subset, geometry, retrieval inputs, target configuration, and so
on.

### Compute Configuration (`compute.toml`) 

The compute configuration file contains settings related to the compute
environment and training recipe. It can be used to configure the accelerator,
devices used, optimizer and learning-rate schedule as well as training duration.

## Logging and Monitoring

Training logs are automatically saved for TensorBoard visualization:

```bash
tensorboard --logdir lightning_logs
```

