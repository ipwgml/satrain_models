"""
satrain_models.xgboost_retrieval
===============================

Provides an implementation of XGBoost for satellite retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


class XGBoostRetrieval:
    """
    XGBoost-based satellite retrieval model.

    A standalone implementation for satellite precipitation retrieval using XGBoost.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize XGBoost retrieval model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
        """

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install it with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.kwargs = kwargs

        # XGBoost parameters
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": max_depth,
            "eta": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "random_state": random_state,
            **kwargs,
        }

        self.model = None
        self._is_fitted = False
        self._input_shape = None

    def _prepare_features(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input features for XGBoost.

        Args:
            x: Input array of shape (batch_size, features) or higher dimensional

        Returns:
            Flattened features array of shape (batch_size, n_features)
        """
        if x.ndim > 2:
            # Flatten higher dimensional data
            batch_size = x.shape[0]
            x_flat = x.reshape(batch_size, -1)
        elif x.ndim == 2:
            # Already in correct format
            x_flat = x
        elif x.ndim == 1:
            # Single sample, reshape to (1, n_features)
            x_flat = x.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return x_flat

    def _prepare_output(self, y: np.ndarray) -> np.ndarray:
        """
        Prepare output for consistent format.

        Args:
            y: Output array from XGBoost

        Returns:
            Output array in consistent format
        """
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
    ) -> "XGBoostRetrieval":
        """
        Fit the XGBoost model.

        Args:
            X: Training features (numpy array)
            y: Training targets (numpy array)
            eval_set: List of (X_val, y_val) tuples for validation
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print training progress

        Returns:
            Self
        """
        # Store input shape for later use
        self._input_shape = X.shape

        # Prepare features and targets
        X_flat = self._prepare_features(X)
        y_flat = y.flatten() if y.ndim > 1 else y

        # Create DMatrix
        dtrain = xgb.DMatrix(X_flat, label=y_flat)

        # Handle validation set
        evals = [(dtrain, "train")]
        if eval_set is not None:
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val_flat = self._prepare_features(X_val)
                y_val_flat = y_val.flatten() if y_val.ndim > 1 else y_val
                dval = xgb.DMatrix(X_val_flat, label=y_val_flat)
                evals.append((dval, f"eval_{i}"))

        # Train model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose,
        )

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted XGBoost model.

        Args:
            X: Input features (numpy array)

        Returns:
            Predictions array
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "Model must be fitted before inference. Call fit() first."
            )

        X_flat = self._prepare_features(X)

        # Make predictions
        dtest = xgb.DMatrix(X_flat)
        y_pred = self.model.predict(dtest)

        return self._prepare_output(y_pred)

    @property
    def num_parameters(self) -> int:
        """Return the approximate number of parameters in the model."""
        if not self._is_fitted or self.model is None:
            return 0

        # XGBoost doesn't have "parameters" in the same way as neural networks
        # We can estimate based on number of trees and max depth
        approx_nodes_per_tree = 2**self.max_depth
        return self.model.num_boosted_rounds() * approx_nodes_per_tree

    @property
    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        # In XGBoost, all parameters are trainable during fitting
        return self.num_parameters

    def save_model(self, path: str) -> None:
        """
        Save the XGBoost model to file.

        Args:
            path: Path to save the model
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before saving.")

        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        """
        Load XGBoost model from file.

        Args:
            path: Path to the saved model
        """
        self.model = xgb.Booster()
        self.model.load_model(path)
        self._is_fitted = True

    def get_feature_importance(
        self, importance_type: str = "weight"
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')

        Returns:
            Dictionary of feature importances
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                "Model must be fitted before getting feature importance."
            )

        return self.model.get_score(importance_type=importance_type)


def create_xgboost(
    n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3, **kwargs
) -> XGBoostRetrieval:
    """
    Create an XGBoost retrieval model.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        **kwargs: Additional XGBoost parameters

    Returns:
        XGBoostRetrieval: XGBoost model instance
    """
    return XGBoostRetrieval(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        **kwargs,
    )
