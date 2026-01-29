"""
satrain_models.bmci
===================

Implements Bayesian Monte-Carlo Integration as a retrieval technique.
"""
from pathlib import Path

from typing import Optional
import numpy as np
import xarray as xr


class BMCI:
    """
    Precipitation retrieval using Bayesian Monte Carlo Integration (BMCI).

    The retrieval assumes independent retrieval errors. To speed up the
    retrieval, an optional cutoff can be applied to discard samples with
    weights lower than a given threshold.

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
            cutoff: If given, samples with a weight smaller than this will be neglected. This can significantly speed
                up the retrieval.
        """
        self.primary_axis = np.argmin(sigma)
        self.Sinv = 1 /  sigma ** 2
        self.cutoff = cutoff
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
        valid = np.where(np.isfinite(x))[0]
        a_p = self.primary_axis
        x = x[valid]
        # Calculate over all samples if not cutoff is given.
        if cutoff is None or a_p not in valid:
            d_x = self.X[:, valid] - x.reshape(1, -1)
            weights = np.exp(
                -1.0 * (d_x * d_x * self.Sinv[None, valid]).sum(axis=-1)
            )
            weights /= weights.sum()
            return (self.y * weights).sum(0)

        c_ind = np.searchsorted(self.X[:, a_p], x[a_p])
        d_x = self.X[c_ind, valid] - x
        base_weight = np.exp(-1.0 * (d_x * self.Sinv[valid] * d_x).sum())

        step = self.X.shape[0] // 100

        # Find lower bound by increasing search bounds.
        lower_ind = max(c_ind - 1000, 0)
        d_x = self.X[lower_ind, valid] - x
        max_weight = np.exp(-1.0 * (d_x[a_p] * self.Sinv[a_p] * d_x[a_p]))
        while (0 < lower_ind) and (cutoff < (max_weight / base_weight)):
            lower_ind = max(lower_ind - 1000, 0)
            d_x = self.X[lower_ind, valid] - x
            max_weight = np.exp(-1.0 * (d_x[a_p] * self.Sinv[a_p] * d_x[a_p]))

        # Find upper bound by increasing search bounds.
        upper_ind = min(c_ind + 1000, self.X.shape[0] - 1)
        d_x = self.X[upper_ind, valid] - x
        max_weight = np.exp(-1.0 * (d_x[a_p] * self.Sinv[a_p] * d_x[a_p]))
        while (upper_ind < self.X.shape[0] - 1) and (cutoff < (max_weight / base_weight)):
            upper_ind = min(upper_ind + 1000, self.X.shape[0] - 1)
            d_x = self.X[upper_ind, valid] - x
            max_weight = np.exp(-1.0 * (d_x[a_p] * self.Sinv[a_p] * d_x[a_p]))

        dx = self.X[lower_ind:upper_ind, valid] - x.reshape(1, -1)
        weights = np.exp(-1.0 * (dx * dx * self.Sinv[None, valid]).sum(-1))
        weights /= weights.sum()
        return (self.y[lower_ind:upper_ind] * weights).sum(0)

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
        sigma = np.sqrt(1.0 / self.Sinv)
        model_data = xr.Dataset({
            "X": (("samples", "features"), self.X),
            "y": (("samples",), self.y),
            "sigma": (("features",), sigma)
        })
        if self.cutoff is not None:
            model_data["cutoff"] = self.cutoff

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


    def predict(self, X: np.array):
        return np.array(
            [self.retrieve_single(x, cutoff=self.cutoff) for x in X]
        )
