"""
satrain_models.bmci
===================

Implements Bayesian Monte-Carlo Integration as a retrieval technique.
"""
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from typing import Optional
import numpy as np
import xarray as xr


LOGGER = logging.getLogger(__name__)


class BMCI:
    """
    Precipitation retrieval using Bayesian Monte Carlo Integration (BMCI).

    The retrieval assumes independent retrieval errors. To speed up the
    retrieval, an optional cutoff can be applied as the number of standard
    deviations along the primary axis to consider, creating deterministic
    summation bounds for numerical stability.

    The retrieval ignores NAN values.
    """
    def __init__(
            self,
            sigma: np.ndarray,
            cutoff: Optional[float] = None
    ):
        """
        Args:
            sigma: A vector containing the observation uncertainties
                for each channels.
            cutoff: If given, number of standard deviations along the primary axis to consider.
                This creates deterministic summation bounds for numerical stability.
        """
        self.primary_axis = np.argmin(sigma)
        self.Sinv = 1 / sigma ** 2
        self.cutoff = cutoff
        if cutoff is not None:
            self.delta_t = cutoff * sigma[self.primary_axis]
        else:
            self.delta_t = None
        self.X = None
        self.y = None


    def fit(self, X, y):
        """
        Fit model.

        Args:
            X: A matrix of shape (m, n) containing m input observation with
                n features.
            y: A vector containing the reference precipitation estimates.
        """
        M, N = X.shape
        assert y.size == M
        assert self.Sinv.size == N

        valid = np.isfinite(X).all(-1)
        X = X[valid]
        y = y[valid]

        inds = np.argsort(X[:, self.primary_axis])
        self.X = X[inds]
        self.y = y[inds]


    def retrieve_single(
            self,
            x: np.array,
            cutoff: Optional[float] = None
    ) -> float:
        """
        Retrieve single observation.

        Args:
            x: A numpy.ndarray containing the observation vector.
            cutoff: Optional cutoff to

        Return:
            The retrieved precipitation value.
        """
        a_p = self.primary_axis
        
        # Check if primary axis is valid
        if not np.isfinite(x[a_p]):
            return np.nan
        
        # Calculate over all samples if no cutoff is given.
        if cutoff is None:
            # Handle NaN values during distance calculation
            valid_mask = np.isfinite(x)
            d_x = self.X[:, valid_mask] - x[valid_mask].reshape(1, -1)
            weights = np.exp(
                -0.5 * (d_x * d_x * self.Sinv[None, valid_mask]).sum(axis=-1)
            )
            weight_sum = weights.sum()
            if weight_sum == 0.0:
                return np.nan
            return (self.y * weights).sum() / weight_sum

        # Use deterministic bounds based on standard deviations along primary axis
        primary_val = x[a_p]
        lower_bound = primary_val - self.delta_t
        upper_bound = primary_val + self.delta_t
        
        # Find indices using binary search
        lower_ind = np.searchsorted(self.X[:, a_p], lower_bound, side='left')
        upper_ind = np.searchsorted(self.X[:, a_p], upper_bound, side='right')
        
        if lower_ind >= upper_ind:
            # No samples in range, return NaN
            return np.nan
        
        # Calculate weights for samples in the determined range
        # Handle NaN values during distance calculation
        valid_mask = np.isfinite(x)
        dx = self.X[lower_ind:upper_ind][:, valid_mask] - x[valid_mask].reshape(1, -1)
        weights = np.exp(-0.5 * (dx * dx * self.Sinv[None, valid_mask]).sum(-1))
        weight_sum = weights.sum()
        
        if weight_sum == 0.0:
            return np.nan
        
        return (self.y[lower_ind:upper_ind] * weights).sum() / weight_sum

    def retrieve_batch(self, X: np.array, cutoff: Optional[float] = None) -> np.array:
        """
        Retrieve multiple observations using memory-efficient vectorized operations.
        
        Args:
            X: A numpy array of shape (batch_size, n_features) containing observations.
            cutoff: Optional cutoff parameter. If None, uses self.cutoff.
            
        Returns:
            Array of retrieved precipitation values.
        """
        if cutoff is None:
            cutoff = self.cutoff
            
        # For cutoff case, fall back to individual processing for now
        if cutoff is not None:
            return np.array([self.retrieve_single(x, cutoff=cutoff) for x in X])
        
        batch_size, n_features = X.shape
        n_samples = self.X.shape[0]
        results = np.zeros(batch_size)
        
        # Process using matrix multiplication instead of broadcasting
        # This avoids creating the huge (batch_size, n_samples, n_features) array
        
        # Handle missing values by masking
        X_masked = X.copy()
        valid_mask = np.isfinite(X)
        
        # For each observation in the batch
        for i in range(batch_size):
            valid_features = valid_mask[i]
            if not valid_features.any():
                results[i] = np.nan
                continue
                
            # Extract valid parts
            x_valid = X[i, valid_features]
            X_train_valid = self.X[:, valid_features]
            Sinv_valid = self.Sinv[valid_features]
            
            # Vectorized distance calculation: (n_samples,)
            diff = X_train_valid - x_valid[None, :]  # (n_samples, n_valid_features)
            d_squared = (diff * diff * Sinv_valid[None, :]).sum(axis=1)  # (n_samples,)
            
            # Calculate weights
            weights = np.exp(-0.5 * d_squared)
            weight_sum = weights.sum()
            
            if weight_sum == 0.0:
                results[i] = np.nan
                continue
                
            # Weighted prediction
            results[i] = (self.y * weights).sum() / weight_sum
        
        return results

    @classmethod
    def load(cls, path: Path) -> "BMCI":
        """
        Load BMCI retrieval.

        Args:
            path: The path to which the model was stored.

        Return:
            The loaded BMCI model.
        """
        data = xr.load_dataset(path)
        sigma = data.sigma.data
        cutoff = None
        if "cutoff" in data:
            cutoff = data.cutoff
        bmci = cls(
            sigma,
            cutoff=cutoff
        )
        # Restore delta_t if saved (for backward compatibility)
        if "delta_t" in data:
            bmci.delta_t = float(data.delta_t.values)
        bmci.fit(
            data.X.data, data.y.data
        )
        return bmci

    def save(self, path: Path) -> Path:
        """
        Save model.

        Saves model into a NetCDF file.

        Args:
            path: The path to which to write the file.

        Return:
            A path object pointin to the stored file.
        """
        if self.X is None or self.y is None:
            raise ValueError("Model must be fitted before saving")
            
        sigma = np.sqrt(1.0 / self.Sinv)
        model_data = xr.Dataset({
            "X": (("samples", "features"), self.X),
            "y": (("samples",), self.y),
            "sigma": (("features",), sigma)
        })
        if self.cutoff is not None:
            model_data["cutoff"] = self.cutoff
        if hasattr(self, 'delta_t') and self.delta_t is not None:
            model_data["delta_t"] = self.delta_t

        model_data["X"].encoding = {
            "dtype": "float32",
            "zlib": True
        }
        model_data["y"].encoding = {
            "dtype": "float32",
            "zlib": True
        }
        model_data["sigma"].encoding = {
            "dtype": "float32",
            "zlib": True
        }
        model_data.to_netcdf(path)
        return path


    def predict(self, X: np.array, n_workers: Optional[int] = None, batch_size: int = 32):
        """
        Predict precipitation for multiple observations.
        
        Args:
            X: Array of observations to retrieve precipitation for.
            n_workers: Number of parallel workers. If None, uses serial processing with batching.
            batch_size: Size of batches for vectorized processing.
        
        Returns:
            Array of precipitation predictions.
        """
        if n_workers is None or n_workers == 1:
            # Use vectorized batch processing instead of Python loop
            results = []
            for i in tqdm(range(0, len(X), batch_size), desc="Processing batches"):
                batch = X[i:i + batch_size]
                batch_results = self.retrieve_batch(batch, cutoff=self.cutoff)
                results.append(batch_results)
            return np.concatenate(results)

        # Parallel processing with batches
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            batch_start = 0
            tasks = []
            while batch_start < X.shape[0]:
                batch_end = batch_start + batch_size
                tasks.append(
                    pool.submit(self.retrieve_batch, X[batch_start:batch_end], self.cutoff)
                )
                batch_start += batch_size
            results = []
            for task in tqdm(tasks, desc="Processing parallel batches"):
                results.append(task.result())

        return np.concatenate(results)
