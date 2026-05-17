"""
satrain_models.lightning
========================

Provides a LightningModule implementing three training recipes.
"""

import logging
import math
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    _LRScheduler,
)
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from satrain_models.metrics import CorrelationCoef, PlotSamples, RelativeBias

# Import for type hints
if typing.TYPE_CHECKING:
    from .config import SatRainConfig

LOGGER = logging.getLogger(__name__)


class KeepAllImprovementsCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that never deletes prior 'best' files.

    Combined with save_top_k set very large, this preserves the full
    history of validation-loss improvements as separate files. Useful
    when later training instability (e.g. BN running-stat corruption
    after an LR restart in fp16) can poison the 'current best' and
    we want to recover an earlier healthy checkpoint.
    """

    def _remove_checkpoint(self, trainer, filepath):  # type: ignore[override]
        return


class BNCollapseDetector(Callback):
    """Stop training if val/loss reads as ~0 for too many consecutive epochs.

    BatchNorm running stats can collapse (typically after an fp16 LR-restart
    spike), making eval-mode outputs degenerate and val/loss read as 0.0.
    Continuing wastes hours; abort early so the next attempt can begin.
    """

    def __init__(self, threshold: float = 1e-6, patience: int = 2):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self._streak = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get("val/loss")
        if metric is None:
            return
        val = float(metric)
        if math.isnan(val) or val <= self.threshold:
            self._streak += 1
            if self._streak >= self.patience:
                LOGGER.error(
                    "val/loss=%.6g for %d consecutive epochs — likely BN collapse; "
                    "stopping training.",
                    val,
                    self._streak,
                )
                trainer.should_stop = True
        else:
            self._streak = 0


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
        loss: nn.Module = nn.MSELoss(),
        approach: str = "adamw_warmup_cosine_annealing",
        learning_rate: Optional[float] = None,
        name: Optional[str] = None,
        satrain_config: Optional[Union[Dict[str, Any], "SatRainConfig"]] = None,
        model_config: Optional[Union[Dict[str, Any], Any]] = None,
    ):
        """
        Args:
            model: A PyTorch model implementing the model to train.
            loss: A loss module defining the loss used to train the model.
            approach: A string specifying the training approach.
            satrain_config: SatRainConfig (or its dict) — embedded in checkpoints
                so any saved .ckpt is self-sufficient as a final model.
            model_config: Architecture config (or its dict), embedded the same way.
        """
        # Coerce configs to plain dicts so save_hyperparameters can serialize
        # them cleanly and downstream loaders don't need the original classes.
        if satrain_config is not None and hasattr(satrain_config, "to_dict"):
            satrain_config = satrain_config.to_dict()
        if model_config is not None and hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])  # keeps logs tidy

        self.model = model
        self.criterion = loss

        valid = {
            "sgd_lr_search",
            "adamw_lr_search",
            "sgd",
            "adamw",
            "sgd_reduce_on_plateau",
            "adamw_reduce_on_plateau",
            "sgd_warmup_cosine_annealing",
            "adamw_warmup_cosine_annealing",
            "sgd_warmup_cosine_annealing_restarts",
            "adamw_warmup_cosine_annealing_restarts",
            "sgd_cosine_annealing_restarts",
            "adamw_cosine_annealing_restarts",
        }
        if approach not in valid:
            raise ValueError(f"Unknown approach '{approach}'. Choose from {valid}.")
        self.approach = approach
        self.learning_rate = learning_rate

        # Initialize validation metrics
        self.val_bias = RelativeBias()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_corrcoef = CorrelationCoef()
        self.plot_samples = PlotSamples()
        self.name = name

    @property
    def experiment_name(self):
        version = 0
        while True:
            if self.name is None:
                name = f"{self.approach}_v{version:02}"
            else:
                name = f"{self.name}_{self.approach}_v{version:02}"
            if not (Path("checkpoints") / (name + ".ckpt")).exists():
                break
            version += 1
        return name

    def _compute_finite_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss only over finite values (no NaN or inf).

        Args:
            pred: Predictions tensor
            y: Target tensor

        Returns:
            Loss computed only over finite values, or zero if no finite values exist
        """
        if y.ndim < pred.ndim:
            y = y.unsqueeze(1)

        # Create mask for finite values in both prediction and target
        mask = torch.isfinite(pred) & torch.isfinite(y)

        # Edge case: entire batch is NaN/inf (no finite values at all).
        # Return a zero loss that is:
        #   1. Connected to the computation graph — so AMP GradScaler can
        #      record inf checks (a detached leaf tensor would trigger
        #      "No inf checks were recorded for this optimizer").
        #   2. Free of NaN — nan_to_num replaces NaN/inf with finite values
        #      first, because NaN * 0.0 = NaN (IEEE 754) which would poison
        #      all model weights via gradient propagation.
        if not mask.any():
            return torch.nan_to_num(pred).sum() * 0.0

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
            y = targets["surface_precip"]
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
        self.log(
            f"train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        # Log LR (first param group) once per step for train
        opt = self.optimizers()
        if opt is not None:
            if isinstance(opt, (list, tuple)):
                lr = opt[0].param_groups[0]["lr"]
            else:
                lr = opt.param_groups[0]["lr"]
            self.log(
                "lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True
            )

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
            y = targets["surface_precip"]
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
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

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
            self.val_bias.update(pred_clean, y_clean)
            self.val_mse.update(pred_clean, y_clean)
            self.val_mae.update(pred_clean, y_clean)
            self.val_corrcoef.update(pred_clean, y_clean)

        # Only works for spatial format
        if pred.ndim == 4:
            self.plot_samples.update(pred, y)

        return loss

    def on_validation_epoch_end(self):
        self.log("val/bias", self.val_bias.compute())
        self.val_bias.reset()
        self.log("val/mae", self.val_mae.compute())
        self.val_mae.reset()
        self.log("val/mse", self.val_mse.compute())
        self.val_mse.reset()
        self.log("val/corrcoef", self.val_corrcoef.compute())
        self.val_corrcoef.reset()
        self.plot_samples.log(self)
        self.plot_samples.reset()

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        """
        Called by lightning to configure optimizers and schedulers.
        """
        hp = self.hparams
        lr = self.learning_rate

        # Learning rate search with SGD
        if hp.approach == "sgd_lr_search":
            LOGGER.info(
                "Performing learning rate search with SGD optimizer across %s steps.",
                self.trainer.estimated_stepping_batches,
            )
            lr = 1e-5
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = StepLR(
                optimizer,
                gamma=1e5 ** (1 / 100),
                step_size=1,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        # Learning rate search with AdamW
        if hp.approach == "adamw_lr_search":
            LOGGER.info(
                "Performing learning rate search with AdamW optimizer across %s steps.",
                self.trainer.estimated_stepping_batches,
            )
            lr = 1e-6
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = StepLR(
                optimizer,
                gamma=1e5 ** (1 / 100),
                step_size=1,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        # SGD, no LR schedule
        if hp.approach == "sgd":
            LOGGER.info("Running training with SGD optimizer, no LR schedule.")
            if lr is None:
                lr = 0.01
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            return optimizer

        # ADAM, no LR schedule
        if hp.approach == "adamw":
            LOGGER.info("Running training with SGD optimizer, no LR schedule.")
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            return optimizer

        # Learning rate search with AdamW
        if hp.approach == "adamw_lr_search":
            LOGGER.info(
                "Performing learning rate search with AdamW optimizer across %s steps.",
                self.trainer.estimated_stepping_batches,
            )
            lr = 1e-6
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = StepLR(
                optimizer,
                gamma=1e5 ** (1 / 100),
                step_size=1,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_reduce_on_plateau":
            if lr is None:
                lr = 0.01
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=5,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_reduce_on_plateau":
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=5,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_warmup_cosine_annealing":
            if lr is None:
                lr = 0.02
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=int(0.1 * self.trainer.estimated_stepping_batches),
                    ),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=int(0.9 * self.trainer.estimated_stepping_batches),
                    ),
                ],
                milestones=[0.1 * self.trainer.estimated_stepping_batches],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_warmup_cosine_annealing":
            if lr is None:
                lr = 4.14e-4
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=int(0.1 * self.trainer.estimated_stepping_batches),
                    ),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=int(0.9 * self.trainer.estimated_stepping_batches),
                    ),
                ],
                milestones=[0.1 * self.trainer.estimated_stepping_batches],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_warmup_cosine_annealing_restarts":
            if lr is None:
                lr = 0.02
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=int(0.1 * self.trainer.estimated_stepping_batches),
                    ),
                    CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=int(0.13 * self.trainer.estimated_stepping_batches),
                        T_mult=2,
                    ),
                ],
                milestones=[0.1 * self.trainer.estimated_stepping_batches],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_warmup_cosine_annealing_restarts":
            if lr is None:
                lr = 4.14e-4
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=int(0.1 * self.trainer.estimated_stepping_batches),
                    ),
                    CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=int(0.13 * self.trainer.estimated_stepping_batches),
                        T_mult=2,
                    ),
                ],
                milestones=[0.1 * self.trainer.estimated_stepping_batches],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_cosine_annealing_restarts":
            if lr is None:
                lr = 0.01
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(0.15 * self.trainer.estimated_stepping_batches),
                T_mult=2,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_cosine_annealing_restarts":
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(0.15 * self.trainer.estimated_stepping_batches),
                T_mult=2,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        raise ValueError("Unknow training approach '%s'.", hp.approach)

    def default_callbacks(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        keep_all_improvements: bool = True,
        bn_collapse_detector: bool = True,
    ) -> List[L.Callback]:
        """
        Returns a sensible EarlyStopping and ModelCheckpoint list.

        Args:
            dirpath: Directory where checkpoints are written. Defaults to
                ``checkpoints/<experiment_name>`` so concurrent runs of
                different experiments don't clash. Pass an explicit path
                to claim a dir atomically before training starts.
            keep_all_improvements: If True (default), every val/loss
                improvement is kept as its own file (never overwritten),
                so a healthy earlier checkpoint can be recovered if
                later training collapses.
            bn_collapse_detector: If True, abort training when val/loss
                reads as ~0 for two consecutive epochs.
        """
        if dirpath is None:
            dirpath = Path("checkpoints") / self.experiment_name

        ckpt_cls = KeepAllImprovementsCheckpoint if keep_all_improvements else ModelCheckpoint
        callbacks: List[L.Callback] = [
            ckpt_cls(
                dirpath=str(dirpath),
                # Lightning replaces '/' with '_' in the rendered filename, so this
                # actually picks up the logged 'val/loss' metric (the previous
                # '{val_loss:.4f}' silently fell back to 0.0000 since no metric
                # named 'val_loss' was logged).
                filename="best_{epoch}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                # save_top_k=1 means Lightning only writes a file when the new
                # epoch beats the current best (true "improvement-only" save).
                # The KeepAllImprovementsCheckpoint subclass no-ops the eviction
                # of the prior best, so it stays on disk — the net effect is a
                # complete history of strictly-improving checkpoints rather than
                # one save per epoch.
                save_top_k=1,
                save_last=True,
                auto_insert_metric_name=False,
            ),
        ]
        callbacks[0].CHECKPOINT_NAME_LAST = "last"

        if bn_collapse_detector:
            callbacks.append(BNCollapseDetector())

        if self.approach in {"sgd_simple", "adamw_simple"}:
            callbacks += [
                EarlyStopping(
                    monitor="val/loss",
                    mode="min",
                    patience=10,
                    min_delta=0.0,
                    verbose=True,
                ),
            ]
        return callbacks

    def get_retrieval_fn(self, satrain_config, compute_config):
        """
        Get retrieval callback function for evaluation.
        """
        if compute_config.accelerator in ["gpu", "cuda"]:
            if compute_config.devices is None:
                device_ind = 0
            else:
                device_ind = compute_config.devices[0]
            device = f"cuda:{device_ind}"
        else:
            device = "cpu"
        dtype = compute_config.dtype
        self.model = self.model.to(device=device, dtype=dtype).eval()

        def retrieval_fn(input_data: xr.Dataset) -> xr.Dataset:
            """
            Run retrieval on input data.
            """
            feature_dim = 0
            if "scan" in input_data.dims:
                spatial_dims = ("scan", "pixel")
            elif "latitude" in input_data.dims:
                spatial_dims = ("latitude", "longitude")
            else:
                spatial_dims = ()

            if "batch" in input_data.dims:
                dims = ("batch",) + spatial_dims
                feature_dim += 1
            else:
                dims = spatial_dims

            inpt = {}
            for name in satrain_config.features:

                inpt_data = torch.tensor(input_data[name].data).to(device, dtype)
                if len(dims) == 1:
                    inpt_data = inpt_data.transpose(0, 1)
                inpt[name] = inpt_data

            inpt = torch.cat(list(inpt.values()), dim=feature_dim)

            with torch.no_grad():
                pred = self.model(inpt)
                if isinstance(pred, torch.Tensor):
                    pred = {"surface_precip": pred}

                results = xr.Dataset()
                if "surface_precip" in pred:
                    results["surface_precip"] = (
                        dims,
                        pred["surface_precip"]
                        .select(feature_dim, 0)
                        .float()
                        .cpu()
                        .numpy(),
                    )
                if "probability_of_precip" in pred:
                    pop = pred["probability_of_precip"].select(feature_dim, 0)
                    pop = torch.sigmoid(pop).cpu().numpy()
                    results["probability_of_precip"] = (dims, pop)
                    precip_flag = self.precip_threshold <= pop
                    results["precip_flag"] = (dims, precip_flag)
                if "probability_of_heavy_precip" in pred:
                    pohp = pred["probability_of_heavy_precip"].select(feature_dim, 0)
                    pohp = torch.sigmoid(pohp).cpu().numpy()
                    results["probability_of_heavy_precip"] = (dims, pohp)
                    heavy_precip_flag = self.heavy_precip_threshold <= pohp
                    results["heavy_precip_flag"] = (dims, heavy_precip_flag)

            return results

        return retrieval_fn

    def save(self, satrain_config: "SatRainConfig", output_path: Path) -> None:
        """
        Save model and configuration to path. The file name is determined using
        the experiment name of the lightning module.

        Args:
            satrain_config: The SatRain config the model was trained with.
            output_path: The path in in which to save the model.
        """
        if not output_path.exists() or not output_path.is_dir():
            raise ValueError(
                "Output path for storing the model should point to an existing "
                "directory."
            )
        path = output_path / f"{self.experiment_name}.pt"
        state = self.model.state_dict()
        torch.save(
            {"state_dict": state, "satrain_config": satrain_config.to_dict()}, path
        )
