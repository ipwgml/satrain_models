# Fully Connected Network for SatRain Dataset

This directory contains a Fully Connected implementation for training on the SatRain dataset using the lightning module and data module from the  `satrain_models` package.


## Overview

This model trains a Fully Connected Network precipitation retrieval on the SatRain dataset using the FullyConnectedNetwork implementation from the `satrain_models.fully_connected` module. 


## Files

- `train.py`: Trains the model.
- `test.py`: Evaluates the model on the SatRain test datasets.
- `compute.toml`: Contains the configuration of the training and compute environment.
- `dataset.toml`: Contains the configuation of the dataset.
- `requirements.txt`: Required packages to run the training.
- `README.md`: This documentation file

## Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```


### Training

The basic training command is:

```bash
python train.py compute.toml --dataset-config dataset.toml
```

The configuration of the Fully connected models and the compute environment are read from the ``dataset.toml`` and ``compute.toml`` files. Therefore no additional parameters are required for the train command.

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

This works for both final models stored in the ``models`` directory as well as checkpoints from the ``checkpoints`` folder. The test results are stored in NetCDF format in the ``test_results`` directory.

## Configuration

The configuration is split into two files for better organization:

### Dataset Configuration (`dataset.toml`)

This file contains the configuration of the SatRain dataset. It is used to
configure the subset, geometry, retrieval inputs, target configuration, and so
on.

### Compute Configuration (`compute.toml`) 

The compute configuration file contains settings related to the compute
environment and training recipe. It can be used to configure the accelerator,
devices used, optimizer and learning-rate schedule as well as training duration.

If not explicitly specified - under [model_config.fully_connected], the model uses a fully connected neural network
with two hidden layers and the following number of neurons per layer:
hidden_layer = [8, 2]

## Logging and Monitoring

Training logs are automatically saved for TensorBoard visualization:

```bash
tensorboard --logdir lightning_logs
```

