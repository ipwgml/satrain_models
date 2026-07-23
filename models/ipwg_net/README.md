# IPWG-Net

This directory contains the tools required to train the IPWG-Net baseline model using custom SatRain input configurations.

## Requirements

Install the `satrain_models` package from the root of this repository:

```bash
pip install -e .
```

## Running the Training

Start a training run by executing:

```bash
python train.py
```

## Configuration

The training setup is controlled through two TOML configuration files:

- `dataset.toml`: Defines the SatRain dataset and input configuration used for training.
- `compute.toml`: Defines training parameters and settings for the compute environment.

Modify these files as needed before launching the training script.
