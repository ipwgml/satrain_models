"""
satrain_models.bmci_fast
========================

Fast C++ implementation of Bayesian Monte-Carlo Integration.
"""
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr

try:
    from .bmci_c import BMCICore, openmp_available, openmp_max_threads
    HAS_C_EXTENSION = True
    HAS_OPENMP = openmp_available()
    MAX_THREADS = openmp_max_threads()
except ImportError:
    HAS_C_EXTENSION = False
    HAS_OPENMP = False
    MAX_THREADS = 1
    BMCICore = None
    openmp_available = None
    openmp_max_threads = None

from .bmci import BMCI

LOGGER = logging.getLogger(__name__)


class BMCIc(BMCI):
    """
    Fast C++ implementation of BMCI using pybind11.
    
    Uses deterministic cutoff bounds based on standard deviations along the
    primary axis for improved numerical stability and performance.
    
    Falls back to pure Python implementation if C extension is not available.
    """
    
    def __init__(self, sigma: np.ndarray, cutoff: Optional[float] = None):
        """
        Args:
            sigma: A vector containing the observation uncertainties for each channel.
            cutoff: If given, number of standard deviations along the primary axis to consider.
                This creates deterministic summation bounds for numerical stability.
        """
        super().__init__(sigma, cutoff)
        
        if not HAS_C_EXTENSION:
            LOGGER.warning(
                "C extension not available, falling back to Python implementation. "
                "Compile the C extension for better performance."
            )
            self._use_c = False
        else:
            self._use_c = True
            cutoff_val = cutoff if cutoff is not None else -1.0
            self._core = BMCICore(sigma.tolist(), cutoff_val)
            
            if HAS_OPENMP:
                LOGGER.info(f"OpenMP enabled with {MAX_THREADS} maximum threads")
            else:
                LOGGER.info("OpenMP not available, using single-threaded C implementation")
    
    def fit(self, X, y):
        """
        Fit model.
        
        Args:
            X: A matrix of shape (m, n) containing m input observations with n features.
            y: A vector containing the reference precipitation estimates.
        """
        if self._use_c:
            # Ensure arrays are contiguous and double precision
            X = np.ascontiguousarray(X, dtype=np.float64)
            y = np.ascontiguousarray(y, dtype=np.float64)
            self._core.fit(X, y)
            # Also call parent to store data for saving
            super().fit(X, y)
        else:
            # Fall back to parent implementation
            super().fit(X, y)
    
    def predict(
            self,
            X: np.array,
            n_workers: Optional[int] = None,
            batch_size: int = 1000,
            use_vectorized: bool = True
    ):
        """
        Predict precipitation for multiple observations.
        
        Args:
            X: Array of observations to retrieve precipitation for.
            n_workers: Number of OpenMP threads for C implementation, or processes for Python.
            batch_size: Size of batches for processing (used only by Python implementation).
            use_vectorized: Use vectorized C implementation when possible (no cutoff case).
        
        Returns:
            Array of precipitation predictions.
        """
        if self._use_c:
            # Use C implementation
            X = np.ascontiguousarray(X, dtype=np.float64)
            n_threads = n_workers if n_workers is not None else -1
            
            if use_vectorized and not self.cutoff:
                return self._core.predict_batch_vectorized(X, n_threads)
            else:
                return self._core.predict_batch(X, n_threads)
        else:
            # Fall back to parent implementation
            return super().predict(X, n_workers, batch_size)
    
    def retrieve_single(self, x: np.array, cutoff: Optional[float] = None) -> float:
        """
        Retrieve single observation.
        
        Args:
            x: A numpy.ndarray containing the observation vector.
            cutoff: Optional cutoff (not used in C implementation, uses instance cutoff).
        
        Return:
            The retrieved precipitation value.
        """
        if self._use_c:
            # For single predictions, use batch predict with size 1
            x_batch = x.reshape(1, -1)
            x_batch = np.ascontiguousarray(x_batch, dtype=np.float64)
            result = self._core.predict_batch(x_batch)
            return result[0]
        else:
            # Fall back to parent implementation
            return super().retrieve_single(x, cutoff)
    
    @classmethod
    def load(cls, path: Path) -> "BMCIc":
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
            cutoff = float(data.cutoff.values)
        
        bmci = cls(sigma, cutoff=cutoff)
        # Restore delta_t if saved (for backward compatibility)
        if "delta_t" in data:
            bmci.delta_t = float(data.delta_t.values)
        bmci.fit(data.X.data, data.y.data)
        return bmci
