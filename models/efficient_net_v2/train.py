#!/usr/bin/env python3
"""
Training script for EfficientNetV2 model on SatRain dataset.
All configuration is read from TOML files.
"""
import logging
from pathlib import Path

import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from satrain_models import (
    SatRainEstimationModule, create_efficient_net_v2, SatRainConfig, 
    SatRainDataModule, tensorboard_to_netcdf,ComputeConfig
)

LOGGER = logging.getLogger("efficient_net_v2_training")

def main():
    """Main training function - all configuration from TOML files."""

    # Load dataset configuration
    dataset_config_path = Path("dataset.toml")
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    config = SatRainConfig.from_toml_file(dataset_config_path)
    LOGGER.info(f"Loaded SatRain config from: {dataset_config_path}")

    # Load compute configuration
    compute_config_path = Path("compute.toml")
    if not compute_config_path.exists():
        raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
    compute_config = ComputeConfig.from_toml_file(compute_config_path)
    LOGGER.info(f"Loaded compute config from: {compute_config_path}")

    # Create data module
    datamodule = SatRainDataModule(
        config=config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
        #spatial=True,  # Use spatial dataset for CNN
    )

    # Create EfficientNetV2 model
    model_size = 'small'  # Options: 'small', 'medium', 'large'
    efficient_net = create_efficient_net_v2(
        n_channels=datamodule.num_features,
        n_outputs=1,
        model_size=model_size,
        pretrained=False,  # we can use pretrained weights if needed
    )

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=efficient_net,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
    )
    experiment_name = lightning_module.experiment_name

    loggers = [TensorBoardLogger(save_dir="lightning_logs", name=f"{experiment_name}_efficient_net_v2_{model_size}")]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=compute_config.max_epochs,
        accelerator=compute_config.accelerator,
        devices=compute_config.devices,
        strategy="auto",
        precision=compute_config.precision,
        logger=loggers,
        callbacks=lightning_module.default_callbacks(),
        log_every_n_steps=compute_config.log_every_n_steps,
        check_val_every_n_epoch=compute_config.check_val_every_n_epoch,
        accumulate_grad_batches=compute_config.accumulate_grad_batches,
    )

    # Train the model
    LOGGER.info(f"Starting the training: {compute_config_path}")
    trainer.fit(lightning_module, datamodule)

    # Extract and save training metrics
    LOGGER.info(f"Training finished. Saving metrics.")
    current_log_dir = loggers[0].log_dir
    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)
    log_path = Path(current_log_dir)
    version_name = log_path.name
    metadata = {
        "experiment": "efficient_net_v2_regression",
        "version": version_name,
        "approach": compute_config.approach,
    }
    output_path = netcdf_dir / (f"{experiment_name}_efficient_net_v2_{model_size}_{version_name}_metrics.nc")
    tensorboard_to_netcdf(
        log_dir=current_log_dir, output_path=output_path, metadata=metadata
    )

    # Save model with dataset config
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)
    lightning_module.save(config, output_path)


if __name__ == "__main__":
    main()

##############################################
########### OLD efficientnetv2 CODE - IGNORE ###########
##############################################
# def main():
#     """Main training function - all configuration from TOML files."""
    
#     # Load dataset configuration
#     dataset_config_path = Path("dataset.toml")
#     if not dataset_config_path.exists():
#         raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
#     config = SatRainConfig.from_toml_file(dataset_config_path)
#     LOGGER.info(f"Loaded dataset config from: {dataset_config_path}")
    
#     # Load compute configuration
#     compute_config_path = Path("compute.toml")
#     if not compute_config_path.exists():
#         raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
#     compute_config = SatRainConfig.from_toml_file(compute_config_path)
#     print(f"✓ Loaded compute config from: {compute_config_path}")

#     # Training settings
#     compute_settings = getattr(compute_config, 'compute', {})
#     max_epochs = compute_settings.get('max_epochs', 100)
#     batch_size = compute_settings.get('batch_size', 8)
#     num_workers = compute_settings.get('num_workers', 4)
#     approach = compute_settings.get('approach', 'adamw_simple')
    
#     # Hardware settings
#     accelerator = compute_settings.get('accelerator', 'gpu')
#     devices = compute_settings.get('devices', 1)
#     precision = compute_settings.get('precision', '32')
    
#     # Logging settings
#     output_dir = compute_settings.get('output_dir', './lightning_logs')

#     # Create data module
#     datamodule = SatRainDataModule(
#         config=config,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=compute_settings.get('pin_memory', True),
#         persistent_workers=compute_settings.get('persistent_workers', True),
#         spatial=True  # Use spatial dataset for CNN
#     )

#     # Create EfficientNetV2 model
#     model_size = 'small'  # Options: 'small', 'medium', 'large'
#     efficient_net = create_efficient_net_v2(
#         n_channels=datamodule.num_features,
#         n_outputs=1,
#         model_size= model_size,
#         pretrained=False,  # we can use pretrained weights if needed
#     )

#     # Create Lightning module
#     lightning_module = SatRainEstimationModule(
#         model=efficient_net,
#         loss=nn.MSELoss(),
#         approach=approach,
#     )
    
#     loggers = [TensorBoardLogger(save_dir=output_dir, name=f"efficient_net_v2_regression_{model_size}")]

#     # Create trainer
#     trainer = L.Trainer(
#         max_epochs=max_epochs,
#         accelerator=accelerator,
#         devices=devices,
#         strategy="auto",
#         precision=precision,
#         logger=loggers,
#         callbacks=lightning_module.default_callbacks(),
#         log_every_n_steps=compute_settings.get('log_every_n_steps', 10),
#         check_val_every_n_epoch=compute_settings.get('check_val_every_n_epoch', 1),
#         accumulate_grad_batches=compute_settings.get('accumulate_grad_batches', 1),
#     )

#     # Train the model
#     trainer.fit(lightning_module, datamodule)

#     # Save metrics to .netcdf
#     current_log_dir = loggers[0].log_dir
#     netcdf_dir = Path("netcdf_metrics")
#     netcdf_dir.mkdir(exist_ok=True)
#     log_path = Path(current_log_dir)
#     version_name = log_path.name
#     output_path = netcdf_dir / f"efficient_net_v2_regression_{model_size}_{version_name}_metrics.nc"
#     metadata = {
#         'experiment': 'efficient_net_v2_regression',
#         'version': version_name,
#         'approach': approach,
#     }

#     # Extract and save metrics
#     tensorboard_to_netcdf(
#         log_dir=current_log_dir,
#         output_path=output_path,
#         metadata=metadata
#     )


# if __name__ == '__main__':
#     main()

##############################################
########### OLD Unet CODE - IGNORE ###########
##############################################

# #!/usr/bin/env python3
# """
# Clean PyTorch Lightning training script for UNet model on SatRain dataset.
# All configuration is read from TOML files.
# """
# from pathlib import Path
# import torch.nn as nn
# import lightning as L
# from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# from satrain_models import (
#     SatRainEstimationModule, create_unet, SatRainConfig, SatRainDataModule,
#     tensorboard_to_netcdf
# )


# def main():
#     """Main training function - all configuration from TOML files."""
    


#     # Load dataset configuration
#     dataset_config_path = Path("dataset.toml")
#     if not dataset_config_path.exists():
#         raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
#     config = SatRainConfig.from_toml_file(dataset_config_path)
#     #LOGGER.info
#     print(f"Loaded dataset config from: {dataset_config_path}")
    
#     # Load compute configuration
#     compute_config_path = Path("compute.toml")
#     if not compute_config_path.exists():
#         raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
#     compute_config = SatRainConfig.from_toml_file(compute_config_path)
#     #LOGGER.info
#     print(f"✓ Loaded compute config from: {compute_config_path}")


#     # Training settings
#     compute_settings = getattr(compute_config, 'compute', {})
#     max_epochs = compute_settings.get('max_epochs', 100)
#     batch_size = compute_settings.get('batch_size', 8)
#     num_workers = compute_settings.get('num_workers', 4)
#     approach = compute_settings.get('approach', 'adamw_simple')
    
#     # Hardware settings
#     accelerator = compute_settings.get('accelerator', 'gpu')
#     devices = compute_settings.get('devices', 1)
#     precision = compute_settings.get('precision', '32')
    
#     # Logging settings
#     output_dir = compute_settings.get('output_dir', './lightning_logs')

#     # Create data module
#     datamodule = SatRainDataModule(
#         config=config,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=compute_settings.get('pin_memory', True),
#         persistent_workers=compute_settings.get('persistent_workers', True),
#         spatial=True  # Use spatial dataset for CNN
#     )

#     # Create model
#     unet_model = create_unet(
#         n_channels=datamodule.num_features, 
#         n_outputs=1, 
#         bilinear=False
#     )

#     # Create Lightning module
#     lightning_module = SatRainEstimationModule(
#         model=unet_model,
#         loss=nn.MSELoss(),
#         approach=approach,
#     )
    
#     loggers = [TensorBoardLogger(save_dir=output_dir, name="basic_unet")]

#     # Create trainer
#     trainer = L.Trainer(
#         max_epochs=max_epochs,
#         accelerator=accelerator,
#         devices=devices,
#         strategy="auto",
#         precision=precision,
#         logger=loggers,
#         callbacks=lightning_module.default_callbacks(),
#         log_every_n_steps=compute_settings.get('log_every_n_steps', 10),
#         check_val_every_n_epoch=compute_settings.get('check_val_every_n_epoch', 1),
#         accumulate_grad_batches=compute_settings.get('accumulate_grad_batches', 1),
#     )
    

#     # Train the model
#     trainer.fit(lightning_module, datamodule)

#     # Save metrics to .netcdf
#     current_log_dir = loggers[0].log_dir
#     netcdf_dir = Path("netcdf_metrics")
#     netcdf_dir.mkdir(exist_ok=True)
#     log_path = Path(current_log_dir)
#     version_name = log_path.name
#     output_path = netcdf_dir / f"basic_unet_{version_name}_metrics.nc"
#     metadata = {
#         'experiment': 'basic_unet',
#         'version': version_name,
#         'approach': approach,
#     }

#     # Extract and save metrics
#     tensorboard_to_netcdf(
#         log_dir=current_log_dir,
#         output_path=output_path,
#         metadata=metadata
#     )


# if __name__ == '__main__':
#     main()