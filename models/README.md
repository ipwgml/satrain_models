# Models Directory

This directory contains individual model implementations for the SatRain dataset.

## Structure

Each model should be placed in its own subdirectory with the following requirements:

- **Folder name**: Use a descriptive name for your model (e.g., `basic_unet`)
- **requirements.txt**: Must be present and conda-compatible
- **``train.py`` and ``test.py``**: The training and testing functionality should be implemented in the
  Python scripts called ``train.py`` and ``test.py``, respectively. The ``test.py`` script should calculate the
  model accuracy on the ``xl`` validation datasets, and all testing domains.
  
## Dataset Configuration

The ``satrain_models`` package provides a ``SatRainConfig`` class that encapsulates the configuration of the SatRain dataset. This configuration can be loaded directly from a .toml file. Models are expected to use this mechanism so they can seamlessly adapt to any input configuration. See the ``dataset.toml`` and ``train.py`` file in the ``basic_unet`` example model for an example how to handle the dataset configuration.


## Utilities for PyTorch models

In addition, the satrain_models package offers a Lightning module
(``SatrainEstimationModule``) that implements a reference training procedure
along with essential logging. Reusing this module helps ensure that PyTorch
model training remains consistent and comparable. See the ``train.py`` script in
the ``basic_unet`` example for an example of how to use the lightning module.

  

## Requirements File Format

The `requirements.txt` file should list all dependencies in a format that can be installed with:

```bash
conda install --file requirements.txt
```

Example `requirements.txt`:
```
numpy>=1.20.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
```


## Example Structure

```
models/
├── my_model/
│   ├── requirements.txt     # Required
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── config.yaml         # Configuration
│   └── README.md           # Model-specific docs
└── another_model/
    ├── requirements.txt     # Required
    └── ...                 # Your implementation
```
