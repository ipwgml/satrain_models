
### 2. README for EfficientNetV2 model  
(save as: `efficientnet_v2/README.md`)

```markdown
# EfficientNetV2 + UNet Decoder for SatRain Dataset

This directory contains an EfficientNetV2 backbone with a UNet-style decoder for precipitation regression on the SatRain dataset.

## Overview

Uses `efficientnet_v2_s` / `m` / `l` from torchvision as encoder + custom UNet-style decoder with skip connections.  
Produces full-resolution spatial precipitation output (regression task).

**Main modifications compared to original EfficientNetV2:**
- Skip connections at 4 different scales (stride 16, 8, 4, 2)
- Multi-stage progressive upsampling
- Feature fusion at each decoder stage
- Preserves spatial details from encoder

## Files

- `train.py`          – Training script
- `test.py`           – Evaluation script
- `compute.toml`      – Compute & training settings
- `dataset.toml`      – Dataset & input/output configuration
- `requirements.txt`  – Required packages
- `README.md`         – This file


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

