"Models for PyMC."

from typing import Any, Optional

import pymc as pm
import pymc_bart as pmb
from pymc.distributions.distribution import DistributionMeta

from .base import ModelBuilder


class LogisticRegression(ModelBuilder):
    """
    Custom PyMC model for logistic regression.

    Parameters
    ----------
    coeff_distribution :    PyMC distribution.
                Prior distribution of coefficient vector.
    distribution_kwargs :   dict.
                Keyword arguments for prior distribution.
    sample_kwargs :   dict.
                Keyword arguments for sampler.

    Examples
    --------
    >>> import numpy as np
    >>> import pymc as pm
    >>> from causalpy.pymc_models import LogisticRegression
    >>>
    >>> X = np.random.rand(10, 10)
    >>> y = np.random.rand(10)
    >>> m = LogisticRegression(
    >>>         coeff_distribution=pm.Cauchy,
    >>>         coeff_distribution_kwargs={"alpha": 0, "beta": 1}
    >>> )
    >>>
    >>> m.fit(X, y)
    """

    def __init__(
        self,
        sample_kwargs: Optional[dict[str, Any]] = None,
        coeff_distribution: DistributionMeta = pm.Normal,
        coeff_distribution_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.coeff_distribution = coeff_distribution
        if coeff_distribution_kwargs is None:
            self.coeff_distribution_kwargs = {"mu": 0, "sigma": 50}
        else:
            self.coeff_distribution_kwargs = coeff_distribution_kwargs

        super().__init__(sample_kwargs)

    def build_model(self, X, y, coords) -> None:
        with self:
            self.add_coords(coords)
            X_ = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            beta = self.coeff_distribution(
                "beta", dims="coeffs", **self.coeff_distribution_kwargs
            )
            mu = pm.Deterministic(
                "mu", pm.math.sigmoid(pm.math.dot(X_, beta)), dims="obs_ind"
            )
            pm.Bernoulli("y_hat", mu, observed=y, dims="obs_ind")

class LinearRegression(ModelBuilder):
    """Custom PyMC model for linear regression"""

    def build_model(self, X, y, coords):
        """Defines the PyMC model"""
        with self:
            self.add_coords(coords)
            X = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            y = pm.MutableData("y", y, dims="obs_ind")
            beta = pm.Normal("beta", 0, 50, dims="coeffs")
            sigma = pm.HalfNormal("sigma", 1)
            mu = pm.Deterministic("mu", pm.math.dot(X, beta), dims="obs_ind")
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")


class BARTRegressor(ModelBuilder):
    """
    Class for building BART based regressors for meta-learners.

    Parameters
    ----------
    m : int.
        Number of trees to fit.
    sigma : float.
        Prior standard deviation.
    sample_kwargs : dict.
        Keyword arguments for sampler.
    """

    def __init__(
        self,
        m: int = 20,
        sigma: float = 1.0,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.m = m
        self.sigma = sigma
        super().__init__(sample_kwargs)

    def build_model(self, X, y, coords=None):
        with self:
            self.add_coords(coords)
            X_ = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            mu = pmb.BART("mu", X_, y, m=self.m, dims="obs_ind")
            pm.Normal("y_hat", mu=mu, sigma=self.sigma, observed=y, dims="obs_ind")


class BARTClassifier(ModelBuilder):
    """
    Class for building BART based models for meta-learners.

    Parameters
    ----------
    m : int.
        Number of trees to fit.
    sample_kwargs : dict.
        Keyword arguments for sampler.
    """

    def __init__(
        self,
        m: int = 20,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.m = m
        super().__init__(sample_kwargs)

    def build_model(self, X, y, coords=None):
        with self:
            self.add_coords(coords)
            X_ = pm.MutableData("X", X, dims=["obs_ind", "coeffs"])
            mu_ = pmb.BART("mu_", X_, y, m=self.m, dims="obs_ind")
            mu = pm.Deterministic("mu", pm.math.sigmoid(mu_), dims="obs_ind")
            pm.Bernoulli("y_hat", mu, observed=y, dims="obs_ind")
