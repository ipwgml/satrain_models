"""
satrain_models.lightning
========================

Provides a LightningModule implementing three training recipes.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import MeanSquaredError, MeanAbsoluteError, CorrcoefCorrCoef

from satrain_models.metrics import CorrelationCoef, PlotSamples


class SatRainEstimationModule(L.LightningModule):
    """
    Lightning module for SatRain precipitation estimation with three training approaches:

    - 'sgd_simple': SGD + optional Nesterov, no scheduler (early stopping via callback)
    - 'adamw_simple': AdamW, no scheduler (early stopping via callback)
    - 'adamw_cosine': AdamW + CosineAnnealingWarmRestarts scheduler

    Logged metrics (per step/epoch):
      - train/loss
      - val/loss, val/mse, val/mae, val/corrcoef
      - lr (current learning rate of the first param group)
    """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        approach: str = "adamw_simple",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])  # keeps logs tidy

        self.model = model
        self.criterion = loss

        valid = {"sgd_simple", "adamw_simple", "adamw_cosine"}
        if approach not in valid:
            raise ValueError(f"Unknown approach '{approach}'. Choose from {valid}.")
        self.approach = approach

        # Initialize validation metrics
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_corrcoef = CorrelationCoef()
        self.plot_samples = PlotSamples()


    def _compute_finite_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Compute loss only over finite values (no NaN or inf).
        
        Args:
            pred: Predictions tensor
            y: Target tensor
            
        Returns:
            Loss computed only over finite values, or zero if no finite values exist
        """
        # Create mask for finite values in both prediction and target
        mask = torch.isfinite(pred) & torch.isfinite(y)
        
        # If no finite values, return zero loss (prevents NaN gradients)
        if not mask.any():
            return torch.tensor(0.0, device=y.device, requires_grad=True)
        
        # Compute loss only over finite values
        pred_masked = pred[mask]
        y_masked = y[mask]
        
        return self.criterion(pred_masked, y_masked)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Calculates loss for single input batch and logs errors.

        Args:
            batch: The input and target data.
            batch_idx: Not used.

        Returns:
            A torch.Tensor containing the loss.
        """
        inputs, targets = batch
        
        # Handle input data (can be stacked tensor or dict)
        if isinstance(inputs, dict):
            # If not stacked, concatenate input channels
            x = torch.cat(list(inputs.values()), dim=1)
        else:
            x = inputs
            
        # Extract target (surface precipitation)
        if isinstance(targets, dict):
            y = targets['surface_precip']
        else:
            y = targets

        # Ensure target is float32 and has the right shape
        y = y.float()
        
        pred = self(x)
        
        # Squeeze output to match target shape if needed
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        
        # Calculate loss only over finite values
        loss = self._compute_finite_loss(pred, y)

        # Log loss
        self.log(f"train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # Log LR (first param group) once per step for train
        opt = self.optimizers()
        if opt is not None:
            if isinstance(opt, (list, tuple)):
                lr = opt[0].param_groups[0]["lr"]
            else:
                lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss


    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with comprehensive metrics computation."""
        inputs, targets = batch
        
        # Handle input data (can be stacked tensor or dict)
        if isinstance(inputs, dict):
            # If not stacked, concatenate input channels
            x = torch.cat(list(inputs.values()), dim=1)
        else:
            x = inputs
            
        # Extract target (surface precipitation)
        if isinstance(targets, dict):
            y = targets['surface_precip']
        else:
            y = targets

        # Ensure target is float32 and has the right shape
        y = y.float()
        
        pred = self(x)
        
        # Squeeze output to match target shape if needed
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        
        # Compute main loss only over finite values
        loss = self._compute_finite_loss(pred, y)

        # Log main loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        # Compute and log validation metrics
        # Flatten tensors for metric computation
        y_flat = y.view(-1)
        pred_flat = pred.view(-1)
        
        # Remove NaN/inf values for metric computation (only use finite values)
        mask = torch.isfinite(y_flat) * torch.isfinite(pred_flat)
        if mask.sum() > 0:
            y_clean = y_flat[mask]
            pred_clean = pred_flat[mask]

            # Update metrics
            self.val_mse.update(pred_clean, y_clean)
            self.val_mae.update(pred_clean, y_clean)
            self.val_corrcoef.update(pred_clean, y_clean)
        self.plot_samples.update(pred, y)

        return loss

    def on_validation_epoch_end(self):
        self.log("val/mae", self.val_mae.compute())
        self.val_mae.reset()
        self.log("val/mse", self.val_mse.compute())
        self.val_mse.reset()
        self.log("val/corrcoef", self.val_corrcoef.compute())
        self.val_corrcoef.reset()
        self.plot_samples.log(self)
        self.plot_samples.reset()

    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        """
        Called by lightning to configure optimizers and schedulers.
        """
        hp = self.hparams

        if hp.approach == "sgd_simple":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=0.1 / 256 * 32,
                momentum=hp.sgd_momentum,
                nesterov=hp.sgd_nesterov,
            )
            return optimizer  # no scheduler; use EarlyStopping callback

        if hp.approach == "adamw_simple":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=1e-3 / 256 * 32,
            )
            return optimizer  # no scheduler; use EarlyStopping callback

        # adamw_cosine
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3 / 256 * 32,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-8,
        )
        # Lightning expects a dict if you want nice logging/interval control
        scheduler_conf = {
            "scheduler": scheduler,
            "interval": "epoch",   # update each epoch; CAWR updates per step internally but stepping each epoch is fine
            "monitor": "val/loss", # NOT required by CAWR, but harmless and consistent with ES
        }
        return [optimizer], [scheduler_conf]

    def default_callbacks(self) -> List[L.Callback]:
        """
        Returns a sensible EarlyStopping and ModelCheckpoint list.
        Use in Trainer(callbacks=SatRainModule.default_callbacks(...))
        """
        callbacks = [
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch}-{val_loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                auto_insert_metric_name=False,
            ),
        ]
        if self.approach in {"sgd_simple", "adamw_simple"}:
            callbacks += [
                EarlyStopping(monitor="val/loss", mode="min", patience=10, min_delta=0.0, verbose=True),
            ]
        return callbacks
