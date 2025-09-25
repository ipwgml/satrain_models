"""
satrain_models.tensorboard_to_netcdf
====================================

Utilities to extract TensorBoard metrics and save them as NetCDF files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    warnings.warn("xarray not available. Install with: pip install xarray netcdf4")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("tensorboard not available. Install with: pip install tensorboard")


def extract_scalars_from_tensorboard(
    log_dir: Union[str, Path], 
    tags: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract scalar metrics from TensorBoard logs.
    
    Args:
        log_dir: Path to TensorBoard log directory (e.g., lightning_logs/basic_unet/version_0)
        tags: List of metric tags to extract. If None, extracts all available scalars
        
    Returns:
        Dictionary mapping tag names to DataFrames with columns: [step, wall_time, value]
    """
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("tensorboard is required. Install with: pip install tensorboard")
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Find the event file
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    # Load events
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()
    
    # Get available scalar tags
    available_tags = event_acc.Tags()['scalars']
    
    # Categorize metrics for better logging
    train_metrics = [tag for tag in available_tags if 'train' in tag.lower()]
    val_metrics = [tag for tag in available_tags if 'val' in tag.lower()]
    other_metrics = [tag for tag in available_tags if 'train' not in tag.lower() and 'val' not in tag.lower()]
    
    print(f"📊 Found TensorBoard metrics in {log_dir.name}:")
    print(f"  Training metrics: {train_metrics}")
    print(f"  Validation metrics: {val_metrics}")
    print(f"  Other metrics: {other_metrics}")
    
    if tags is None:
        # Extract all available tags, but prioritize important metrics
        tags = available_tags
        
        # Check for specific validation metrics we expect
        expected_val_metrics = ['val/loss', 'val/mse', 'val/mae', 'val/pearson']
        missing_val_metrics = [m for m in expected_val_metrics if m not in available_tags]
        if missing_val_metrics and val_metrics:
            print(f"  ⚠ Expected validation metrics not found: {missing_val_metrics}")
        elif not val_metrics:
            print(f"  ⚠ No validation metrics found (likely fast_dev_run was used)")
    else:
        # Filter to only available tags
        tags = [tag for tag in tags if tag in available_tags]
        missing_tags = [tag for tag in tags if tag not in available_tags]
        if missing_tags:
            print(f"  ⚠ Requested tags not found: {missing_tags}")
    
    # Extract scalar data
    scalar_data = {}
    for tag in tags:
        scalar_events = event_acc.Scalars(tag)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'step': event.step,
                'wall_time': event.wall_time, 
                'value': event.value
            }
            for event in scalar_events
        ])
        
        scalar_data[tag] = df
        print(f"  ✓ Extracted {tag}: {len(df)} data points")
    
    return scalar_data


def scalars_to_netcdf(
    scalar_data: Dict[str, pd.DataFrame],
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Convert TensorBoard scalar data to NetCDF format using xarray.
    
    Args:
        scalar_data: Dictionary of scalar data from extract_scalars_from_tensorboard
        output_path: Path to save NetCDF file
        metadata: Optional metadata to include in NetCDF attributes
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("xarray is required. Install with: pip install xarray netcdf4")
    
    if not scalar_data:
        raise ValueError("No scalar data provided")
    
    # Find the maximum number of steps across all metrics
    all_steps = set()
    for df in scalar_data.values():
        all_steps.update(df['step'].values)
    
    max_steps = sorted(all_steps)
    
    # Create xarray Dataset
    data_vars = {}
    coords = {'step': max_steps}
    
    for tag, df in scalar_data.items():
        # Create a full array filled with NaN
        values = np.full(len(max_steps), np.nan, dtype=np.float32)
        wall_times = np.full(len(max_steps), np.nan, dtype=np.float64)
        
        # Fill in actual values
        for _, row in df.iterrows():
            step_idx = max_steps.index(row['step'])
            values[step_idx] = row['value']
            wall_times[step_idx] = row['wall_time']
        
        # Clean tag name for NetCDF (replace problematic characters)
        clean_tag = tag.replace('/', '_').replace(' ', '_')
        
        data_vars[clean_tag] = (['step'], values)
        data_vars[f'{clean_tag}_wall_time'] = (['step'], wall_times)
    
    # Create Dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add metadata
    if metadata:
        ds.attrs.update(metadata)
    
    # Add standard attributes
    ds.attrs.update({
        'title': 'TensorBoard Metrics',
        'description': 'Metrics extracted from TensorBoard logs',
        'created_by': 'satrain_models.tensorboard_to_netcdf',
    })
    
    # Add variable descriptions
    for tag in scalar_data.keys():
        clean_tag = tag.replace('/', '_').replace(' ', '_')
        if clean_tag in ds.data_vars:
            ds[clean_tag].attrs['long_name'] = tag
            ds[clean_tag].attrs['description'] = f'TensorBoard scalar metric: {tag}'
        
        wall_time_var = f'{clean_tag}_wall_time'
        if wall_time_var in ds.data_vars:
            ds[wall_time_var].attrs['long_name'] = f'{tag} wall time'
            ds[wall_time_var].attrs['description'] = f'Wall clock time for {tag}' 
            ds[wall_time_var].attrs['units'] = 'seconds since 1970-01-01T00:00:00'
    
    # Save to NetCDF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ds.to_netcdf(output_path, format='NETCDF4')
    print(f"✓ Saved TensorBoard metrics to NetCDF: {output_path}")


def tensorboard_to_netcdf(
    log_dir: Union[str, Path],
    output_path: Union[str, Path],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Extract TensorBoard scalars and save as NetCDF in one step.
    
    Args:
        log_dir: Path to TensorBoard log directory
        output_path: Path to save NetCDF file
        tags: List of metric tags to extract. If None, extracts all available scalars
        metadata: Optional metadata to include in NetCDF attributes
    """
    # Extract scalar data
    scalar_data = extract_scalars_from_tensorboard(log_dir, tags=tags)
    
    # Convert to NetCDF
    scalars_to_netcdf(scalar_data, output_path, metadata=metadata)


def extract_all_training_runs(
    lightning_logs_dir: Union[str, Path] = "lightning_logs",
    output_dir: Union[str, Path] = "netcdf_metrics",
    experiment_name: str = "basic_unet"
) -> List[Path]:
    """
    Extract metrics from all training runs and save as separate NetCDF files.
    
    Args:
        lightning_logs_dir: Root lightning logs directory
        output_dir: Directory to save NetCDF files
        experiment_name: Name of the experiment (subdirectory in lightning_logs)
        
    Returns:
        List of paths to created NetCDF files
    """
    lightning_logs_dir = Path(lightning_logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_dir = lightning_logs_dir / experiment_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Find all version directories
    version_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith('version_')]
    version_dirs.sort(key=lambda x: int(x.name.split('_')[1]))  # Sort by version number
    
    created_files = []
    
    for version_dir in version_dirs:
        version_name = version_dir.name
        output_path = output_dir / f"{experiment_name}_{version_name}_metrics.nc"
        
        try:
            # Extract metadata from hparams if available
            metadata = {
                'experiment': experiment_name,
                'version': version_name,
                'source_log_dir': str(version_dir)
            }
            
            # Try to read hparams.yaml for additional metadata
            hparams_file = version_dir / "hparams.yaml"
            if hparams_file.exists():
                try:
                    import yaml
                    with open(hparams_file) as f:
                        hparams = yaml.safe_load(f)
                    # Add key hyperparameters to metadata
                    for key in ['lr', 'approach', 'batch_size']:
                        if key in hparams:
                            metadata[f'hparam_{key}'] = hparams[key]
                except ImportError:
                    pass  # yaml not available
                except Exception as e:
                    print(f"⚠ Could not read hparams from {hparams_file}: {e}")
            
            tensorboard_to_netcdf(version_dir, output_path, metadata=metadata)
            created_files.append(output_path)
            
        except Exception as e:
            print(f"⚠ Failed to extract metrics from {version_dir}: {e}")
    
    print(f"✓ Extracted metrics from {len(created_files)} training runs")
    return created_files


if __name__ == "__main__":
    # Example usage
    extract_all_training_runs()
