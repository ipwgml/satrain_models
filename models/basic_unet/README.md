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

Or using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Basic training command:

```bash
python train.py --model-config model.toml --input-config input.toml --output-dir ./outputs
```

Additional training options:

```bash
# Resume from checkpoint
python train.py --model-config model.toml --input-config input.toml --resume ./outputs/checkpoints/latest.pth

# Use Weights & Biases logging
python train.py --model-config model.toml --input-config input.toml --use-wandb
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model-config model.toml --input-config input.toml --checkpoint ./outputs/best_model.pth --output-dir ./evaluation_results
```

Evaluation options:

```bash
# Evaluate on validation set
python evaluate.py --model-config model.toml --input-config input.toml --checkpoint ./outputs/best_model.pth --split validation

# Save more sample predictions
python evaluate.py --model-config model.toml --input-config input.toml --checkpoint ./outputs/best_model.pth --save-samples 10
```

## Configuration

The configuration is split into two files for better organization:

### Model Configuration (`model.toml`)
Contains model architecture, training, and hardware settings:

### Input Configuration (`input.toml`) 
Contains dataset and input source settings with detailed retrieval input tables:

**Model Configuration (`model.toml`):**
- `[model]`: Architecture settings (channels, outputs, bilinear upsampling)
- `[training]`: Training parameters (batch size, epochs, loss function)
- `[training.optimizer]`: Optimizer settings (type, learning rate, weight decay)
- `[training.scheduler]`: Learning rate scheduler configuration
- `[logging]`: Logging and checkpointing intervals
- `[hardware]`: Device configuration

**Input Configuration (`input.toml`):**
- `[dataset]`: Basic dataset settings (sensor, geometry, subset)
- `[target_config]`: Data quality filtering parameters
- `[inputs.{source}]`: Individual input source configurations
  - `[inputs.gmi]`: GMI microwave observations
  - `[inputs.atms]`: ATMS microwave observations
  - `[inputs.geo]`: Geostationary visible/NIR data
  - `[inputs.geo_ir]`: Geostationary infrared data
  - `[inputs.ancillary]`: Meteorological ancillary data
- `[active_inputs]`: Enable/disable input sources
- `[combination]`: Input combination and stacking settings

## Output Structure

Training creates the following output structure:

```
outputs/
├── logs/
│   └── training.log          # Training log file
├── checkpoints/
│   ├── checkpoint_epoch_N.pth # Per-epoch checkpoints
│   └── latest.pth            # Latest checkpoint
├── tensorboard/              # TensorBoard logs
└── best_model.pth           # Best model weights
```

Evaluation creates:

```
evaluation_results/
├── evaluation_metrics.txt    # Human-readable metrics
├── evaluation_metrics.toml   # Machine-readable metrics
├── scatter_plot.png         # Predictions vs ground truth
└── samples/                 # Sample prediction images
    ├── sample_1.png
    ├── sample_2.png
    └── ...
```

## Metrics

The evaluation script computes the following metrics:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Correlation**: Pearson correlation coefficient

## Logging and Monitoring

### TensorBoard
Training logs are automatically saved for TensorBoard visualization:

```bash
tensorboard --logdir ./outputs/tensorboard
```

### Weights & Biases
Enable W&B logging with the `--use-wandb` flag:

```bash
python train.py --config config.yaml --use-wandb
```

## Customization

### Adding New Loss Functions
Edit the `create_loss_function()` function in `train.py`:

```python
def create_loss_function(config):
    loss_type = config['training']['loss']
    
    if loss_type == 'custom':
        return CustomLoss()
    # ... existing loss functions
```

### Adding New Optimizers
Edit the `create_optimizer()` function in `train.py`:

```python
def create_optimizer(model, config):
    optimizer_type = optimizer_config['type']
    
    if optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), ...)
    # ... existing optimizers
```

### Input Source Configuration
The `input.toml` file provides granular control over each input source:

**Example input source configuration:**
```toml
[inputs.gmi]
name = "gmi"
channels = []                   # Empty = use all channels
include_angles = true           # Include viewing angles
normalize = "standardize"       # Normalization method
nan_value = -999.0              # NaN replacement value
```

**Active inputs control:**
```toml
[active_inputs]
gmi = true
atms = false
geo_ir = true
ancillary = true
```

## Hardware Requirements

- **GPU**: Recommended for training (automatically detected)
- **Memory**: Depends on batch size and input image dimensions
- **Storage**: For checkpoints, logs, and dataset

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `model.toml`
2. **Dataset not found**: Check `data_path` in `input.toml`
3. **Input dimension mismatch**: Update `n_channels` in `model.toml` to match active inputs
3. **Import errors**: Ensure all packages are installed correctly

### Performance Tips

1. **Use mixed precision**: Add AMP support for faster training
2. **Increase num_workers**: For faster data loading
3. **Use SSD storage**: For faster dataset access
4. **Monitor GPU utilization**: Ensure efficient resource usage

## Example Workflow

1. **Prepare data**: Ensure SatRain dataset is accessible
2. **Configure**: Modify `model.toml` and `input.toml` for your setup
3. **Install dependencies**: `conda install --file requirements.txt`
4. **Train model**: `python train.py --model-config model.toml --input-config input.toml`
5. **Evaluate results**: `python evaluate.py --model-config model.toml --input-config input.toml --checkpoint ./outputs/best_model.pth`
6. **Analyze outputs**: Review metrics and visualizations

## References

- [SatRain Dataset](https://github.com/ipwgml/satrain)
- [UNet Paper](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/docs/)
