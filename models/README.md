# Models Directory

This directory contains individual model implementations for the SatRain dataset.

## Structure

Each model should be placed in its own subdirectory with the following requirements:

- **Folder name**: Use a descriptive name for your model (e.g., `resnet50`, `transformer_baseline`, `custom_cnn`)
- **requirements.txt**: Must be present and conda-compatible
- **Organization**: You're free to organize the rest of the folder as needed

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

## Model Implementation Guidelines

While you have full freedom in organizing your model folder, consider including:

- Training scripts
- Model definition files
- Evaluation scripts
- Configuration files
- Documentation specific to your model
- Example usage or demo scripts

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