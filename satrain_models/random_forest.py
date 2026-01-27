"""
satrain_models.random_forest
============================

Provides an implementation of Random Forest for satellite retrieval using scikit-learn.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    joblib = None


class RandomForestRetrieval:
    """
    Random Forest-based satellite retrieval model.

    A standalone implementation for satellite precipitation retrieval using scikit-learn's Random Forest.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Random Forest retrieval model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees (None for unlimited)
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            bootstrap: Whether bootstrap samples are used when building trees
            random_state: Random seed
            n_jobs: Number of jobs to run in parallel (None for 1, -1 for all processors)
            **kwargs: Additional RandomForest parameters
        """

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Please install it with: pip install scikit-learn"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        # Create Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        self._is_fitted = False
        self._input_shape = None

    def _prepare_features(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input features for Random Forest.

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
            y: Output array from Random Forest

        Returns:
            Output array in consistent format
        """
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "RandomForestRetrieval":
        """
        Fit the Random Forest model.

        Args:
            X: Training features (numpy array)
            y: Training targets (numpy array)
            sample_weight: Sample weights (optional)
            X_val: Validation features (for logging only, RF doesn't use early stopping)
            y_val: Validation targets (for logging only)
            verbose: Whether to print training progress

        Returns:
            Self
        """
        # Store input shape for later use
        self._input_shape = X.shape

        # Prepare features and targets
        X_flat = self._prepare_features(X)
        y_flat = y.flatten() if y.ndim > 1 else y

        if verbose:
            print(
                f"Training Random Forest with {X_flat.shape[0]} samples and {X_flat.shape[1]} features..."
            )

        # Train model
        self.model.fit(X_flat, y_flat, sample_weight=sample_weight)

        if verbose:
            # Calculate training metrics
            train_pred = self.model.predict(X_flat)
            train_mse = mean_squared_error(y_flat, train_pred)
            train_r2 = r2_score(y_flat, train_pred)
            print(f"Training completed - MSE: {train_mse:.6f}, R²: {train_r2:.6f}")

            # Calculate validation metrics if provided
            if X_val is not None and y_val is not None:
                X_val_flat = self._prepare_features(X_val)
                y_val_flat = y_val.flatten() if y_val.ndim > 1 else y_val
                val_pred = self.model.predict(X_val_flat)
                val_mse = mean_squared_error(y_val_flat, val_pred)
                val_r2 = r2_score(y_val_flat, val_pred)
                print(f"Validation metrics - MSE: {val_mse:.6f}, R²: {val_r2:.6f}")

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest model.

        Args:
            X: Input features (numpy array)

        Returns:
            Predictions array
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before inference. Call fit() first."
            )

        X_flat = self._prepare_features(X)

        # Make predictions
        y_pred = self.model.predict(X_flat)

        return self._prepare_output(y_pred)

    @property
    def num_parameters(self) -> int:
        """Return the approximate number of parameters in the model."""
        if not self._is_fitted:
            return 0

        # Estimate parameters based on number of trees and nodes
        # This is an approximation since Random Forest doesn't have "parameters" like neural networks
        total_nodes = 0
        for tree in self.model.estimators_:
            total_nodes += tree.tree_.node_count

        return total_nodes

    @property
    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        # In Random Forest, all parameters are trainable during fitting
        return self.num_parameters

    def save_model(self, path: str) -> None:
        """
        Save the Random Forest model to file using joblib.

        Args:
            path: Path to save the model
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving.")

        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        """
        Load Random Forest model from file.

        Args:
            path: Path to the saved model
        """
        self.model = joblib.load(path)
        self._is_fitted = True

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of feature importances
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before getting feature importance."
            )

        return self.model.feature_importances_

    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if bootstrap=True and oob_score=True during initialization.

        Returns:
            OOB score or None if not available
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting OOB score.")

        return getattr(self.model, "oob_score_", None)


def create_random_forest_retrieval(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    max_features: Union[str, int, float] = "sqrt",
    **kwargs,
) -> RandomForestRetrieval:
    """
    Create a Random Forest retrieval model.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        max_features: Number of features to consider when looking for the best split
        **kwargs: Additional Random Forest parameters

    Returns:
        RandomForestRetrieval: Random Forest model instance
    """
    return RandomForestRetrieval(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        **kwargs,
    )
