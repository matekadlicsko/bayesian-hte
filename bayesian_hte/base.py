"Base classes."

from typing import Any, Dict, Optional, Self, Tuple, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from arviz import r2_score
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_consistent_length
from xarray import DataArray

from .summary import Summary
from .utils import check_binary


class ModelBuilder(pm.Model):
    """
    This is a wrapper around pm.Model to give scikit-learn like API
    """

    def __init__(self, sample_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.idata = None
        self.sample_kwargs = sample_kwargs if sample_kwargs is not None else {}

        self.X = None
        self.y = None
    
    def __str__(self):
        return f"{self.__class__.__name__}()"

    def build_model(self, X, y, coords) -> None:
        """Build the model.

        Example
        -------
        >>> class CausalPyModel(ModelBuilder):
        >>>    def build_model(self, X, y):
        >>>        with self:
        >>>            X_ = pm.MutableData(name="X", value=X)
        >>>            y_ = pm.MutableData(name="y", value=y)
        >>>            beta = pm.Normal("beta", mu=0, sigma=1, shape=X_.shape[1])
        >>>            sigma = pm.HalfNormal("sigma", sigma=1)
        >>>            mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
        >>>            pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_)
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def _data_setter(self, X) -> None:
        with self:
            pm.set_data({"X": X})

    def fit(self, X, y, coords: Optional[Dict[str, Any]] = None) -> None:
        """Draw samples from posterior, prior predictive, and posterior predictive
        distributions.
        """
        self.build_model(X, y, coords)

        self.X = X
        self.y = y

        with self:
            self.idata = pm.sample(**self.sample_kwargs)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(
                pm.sample_posterior_predictive(self.idata, progressbar=False)
            )
        return self.idata

    def predict(self, X):
        """Predict data given input data `X`"""
        self._data_setter(X)
        with self:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata, var_names=["y_hat", "mu"], progressbar=False
            )
        return post_pred

    def score(self, X=None, y=None) -> pd.Series:
        """Score the Bayesian :math:`R^2` given inputs ``X`` and outputs ``y``.

        .. caution::

            The Bayesian :math:`R^2` is not the same as the traditional coefficient of
            determination, https://en.wikipedia.org/wiki/Coefficient_of_determination.

        """
        X = X or self.X
        y = y or self.y

        yhat = self.predict(X)
        yhat = az.extract(
            yhat, group="posterior_predictive", var_names="y_hat"
        ).T.values
        # Note: First argument must be a 1D array
        return r2_score(y, yhat)


class MetaLearner:
    "Base class for meta-learners."

    def __init__(self) -> None:
        self.cate = None
        self.models = {}

        self.X = None
        self.y = None
        self.treated = None

        self.index = None
        self.labels = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Self:
        """
        Fits base-learners.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y :     pandas.Series of shape (n_samples, ).
                Target vector.
        treated :   pandas.Series of shape (n_samples, ).
        """
        check_consistent_length(X, y, treated)
        check_binary(treated)

        self.X = X
        self.y = y
        self.treated = treated

        if isinstance(X, pd.DataFrame):
            self.index = X.index
            self.labels = X.columns
        else:
            self.index = np.arange(X.shape[0])
            self.labels = np.arange(X.shape[1])

        self._fit(X, y, treated)

        self.cate = self.predict_cate(X)
        return self

    def _fit(self, X, y, treated):
        """_fit method for base-learners.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y :     pandas.Series of shape (n_samples, ).
                Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.

        Returns
        -------
        Self
            Fitted meta-learner.
        """
        raise NotImplementedError()

    def fit_learner(
        self, model: Union[ModelBuilder, BaseEstimator], X: pd.DataFrame, y: pd.Series
    ) -> Union[ModelBuilder, BaseEstimator]:
        """
        Fits a single learner.

        Parameters
        ----------
        model : Union[ModelBuilder, BaseEstimator].
            Model to fit.
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y :     pandas.Series of shape (n_samples, ).
                Target vector.
        """
        raise NotImplementedError()

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict out-of-sample conditional average treatment effect on given input X.
        For in-sample treatement effect self.cate should be used.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_featues).
        """
        raise NotImplementedError()

    def predict_ate(self, X: pd.DataFrame) -> np.float64:
        """
        Predict out-of-sample average treatment effect on given input X. For in-sample
        treatement effect self.ate() should be used.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_featues).
        """
        return self.predict_cate(X).mean()

    def ate(self) -> np.float64:
        "Returns in-sample average treatement effect."
        return self.cate.mean()


class FrequentistMetaLearner(MetaLearner):
    "Base class for sklearn based meta-learners."

    def fit_learner(self, model, X, y):
        return model.fit(X, y)

    def bootstrap(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        n_iter: int = 1000,
    ) -> np.array:
        """
        Runs bootstrap n_iter times on a sample of size n_samples.
        Fits on (X_ins, y, treated), then predicts on X.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform.
        """
        if X is None:
            X = X_ins

        # Bootstraping overwrites these attributes
        models, cate = self.models, self.cate

        # Calculate number of treated and untreated data points
        n1 = self.treated.sum()
        n0 = self.treated.count() - n1

        # Prescribed treatement variable of samples
        t_bs = pd.Series(n0 * [0] + n1 * [1], name="treatement")

        results = []

        for _ in range(n_iter):
            # Take sample with replacement from our data in a way that we have
            # the same number of treated and untreated data points as in the whole
            # data set.
            X0_bs = X_ins[treated == 0].sample(n=n0, replace=True)
            y0_bs = y.loc[X0_bs.index]

            X1_bs = X_ins[treated == 1].sample(n=n1, replace=True)
            y1_bs = y.loc[X1_bs.index]

            X_bs = pd.concat([X0_bs, X1_bs], axis=0).reset_index(drop=True)
            y_bs = pd.concat([y0_bs, y1_bs], axis=0).reset_index(drop=True)

            self.fit(X_bs.reset_index(drop=True), y_bs, t_bs)
            results.append(self.predict_cate(X))

        self.models = models
        self.cate = cate
        return np.array(results)

    def ate_confidence_interval(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        q: float = 0.05,
        n_iter: int = 1000,
    ) -> tuple:
        """Estimates confidence intervals for ATE on X using bootstraping.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        q : float, default=.05.
            Quantile to compute. Should be between in the interval [0, 1].
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform.
        """
        cates = self.bootstrap(X_ins, y, treated, X, n_iter)
        ates = cates.mean(axis=0)
        return np.quantile(ates, q=q / 2), np.quantile(cates, q=1 - q / 2)

    def cate_confidence_interval(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        q: float = 0.05,
        n_iter: int = 1000,
    ) -> np.array:
        """Estimates confidence intervals for CATE on X using bootstraping.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        q : float, default=.05.
            Quantile to compute. Should be between in the interval [0, 1].
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform."""
        cates = self.bootstrap(X_ins, y, treated, X, n_iter)
        conf_ints = np.append(
            np.quantile(cates, q / 2, axis=0).reshape(-1, 1),
            np.quantile(cates, 1 - q / 2, axis=0).reshape(-1, 1),
            axis=1,
        )
        return conf_ints

    def bias(
        self,
        X_ins: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
        X: pd.DataFrame = None,
        n_iter: int = 1000,
    ) -> np.float64:
        """Calculates bootstrap estimate of bias of CATE estimator.

        Parameters
        ----------
        X_ins : pandas.DataFrame of shape (n_samples, n_featues).
                Training data.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        q : float, default=.05.
            Quantile to compute. Should be between in the interval [0, 1].
        n_iter : int, default = 1000.
            Number of bootstrap iterations to perform."""
        if X is None:
            X = X_ins

        pred = self.predict_cate(X=X)
        bs_pred = self.bootstrap(X_ins, y, treated, X, n_iter).mean(axis=0)

        return (bs_pred - pred).mean()

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        """
        Returns a dictionary of R^2 scores of base-learners, mean accuracy in case of
        propensity score estimator.

        Parameters
        ----------
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        y : pandas.Series of shape (n_samples, ).
            Target vector.
        treated :   pandas.Series of shape (n_samples, ).
                    Treatement assignment indicator consisting of zeros and ones.
        """
        raise NotImplementedError()

    def average_uplift_by_quantile(
        self,
        X: pd.DataFrame,
        nbins: int = 20,
    ) -> pd.DataFrame:
        """
        Returns average uplift in quantile groups.

        Parameters
        ----------
        X : pandas.DataFrame of shape (_, n_features).
            Data to predict on.
        nbins : int.
            Number of bins.
        """
        preds = self.predict_cate(X)
        nunique = preds.unique().shape[0]

        uplift = (
            pd.DataFrame(
                {
                    "Mean uplift by quantile": preds,
                    "quantile": pd.qcut(preds, q=min(nunique, nbins)),
                }
            )
            .groupby("quantile")
            .mean()
            .sort_values("quantile", ascending=False)
        )

        return uplift

    def summary(self, n_iter: int = 100) -> Summary:
        """
        Returns.

        Parameters
        ----------
        n_iter :    int, default=1000.
                    Number of bootstrap iterations to perform.
        """
        # Bootstrapping confidence intervals for ATE
        bootstrapped_cates = self.bootstrap(
            self.X, self.y, self.treated, self.X, n_iter
        )
        bootstrapped_ates = bootstrapped_cates.mean(axis=1)
        conf_ints = (
            np.quantile(bootstrapped_ates, q=0.025),
            np.quantile(bootstrapped_ates, q=0.975),
        )
        conf_ints = map(lambda x: round(x, 2), conf_ints)

        # Calculate bias
        cates = self.predict_cate(self.X)
        ate = round(cates.mean(), 2)

        bias = (bootstrapped_cates.mean(axis=0) - cates).mean()
        bias = round(bias, 2)

        score = self.score(self.X, self.y, self.treated)
        models = {
            k: [type(v).__name__, round(score[k], 2)] for k, v in self.models.items()
        }

        s = Summary()
        s.add_title(["Conditional Average Treatment Effect Estimator Summary"])
        s.add_row("Number of observations", [self.index.shape[0]], 2)
        s.add_row("Number of treated observations", [self.treated.sum()], 2)
        s.add_row("Average treatement effect (ATE)", [ate], 2)
        s.add_row("95% Confidence interval for ATE", [tuple(conf_ints)], 1)
        s.add_row("Estimated bias", [bias], 2)
        s.add_title(["Base learners"])
        s.add_header(["", "Model", "R^2"], 1)

        for name, x in models.items():
            s.add_row(name, x, 1)

        return s

    def plot(
        self,
        nbins: int = 20,
        figsize: tuple[int] = (20, 20),
        fontsize: int = 40,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots average uplift per decile and cummulative uplift curve.

        Parameters
        ----------
        nbins : int, default=20.
            Number of bins.
        figsize: tuple[int], default=(20, 20).
            Size of figure.
        fontsize: int, default=40.
            Size of font to use.
        """
        fig, ax = plt.subplots(2, 1, figsize=figsize)

        uplift = self.average_uplift_by_quantile(self.X, nbins=nbins).sort_values(
            by="Mean uplift by quantile", ascending=False
        )

        # nbins might change here if nbins is too big
        nbins = uplift.shape[0]

        uplift.plot.bar(ax=ax[0])
        ax[0].set_title("Uplift by quantile", fontsize=fontsize)

        (
            uplift.cumsum()
            .reset_index(drop=True)
            .set_index(np.linspace(0, 100, nbins))
            .plot(ax=ax[1], xlabel="Percentage of population", ylabel="Uplift")
        )
        ax[1].set_title("Cumulative uplift", fontsize=fontsize)

        return fig, ax


class BayesianMetaLearner(MetaLearner):
    "Base class for PyMC based meta-learners."

    def fit_learner(self, model, X, y):
        return model.fit(X, y, coords={"coeffs": X.columns, "obs_ind": X.index})

    def ate_hdi(self, X) -> np.array:
        """
        Estimates high density interval of average treatement effect on X.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Feature matrix.
        """
        cate = self.predict_cate(X)
        hdi = az.hdi(cate.mean(dim="obs_ind")).mu.values
        return hdi

    def cate_hdi(self, X) -> np.array:
        """
        Estimates high density interval of conditional average treatement effect on X.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Feature matrix.
        """
        cate = self.predict_cate(X)
        hdi = az.hdi(cate).mu.values
        return hdi

    def summary(self) -> Summary:
        "Returns summary."
        hdi = self.ate_hdi(self.X)
        ate = self.ate()

        s = Summary()

        s.add_title(["Conditional Average Treatment Effect Estimator Summary"])
        s.add_row("Number of observations", [self.index.shape[0]], 2)
        s.add_row("Number of treated observations", [self.treated.sum()], 2)
        s.add_row("Average treatement effect (ATE)", [ate.item()], 2)
        s.add_row("HDI for ATE", [tuple(hdi)], 1)
        s.add_title(["Base learners"])
        # s.add_header(["", "Model", "R^2"], 1)

        # for name, model in self.models.items():
        #     s.add_row(name, [ model, model.score() ], 1)

        return s

    def predict_cate(self, X: pd.DataFrame) -> DataArray:
        """
        Predicts distribution of treatement effect.

        Parameters
        ----------
        X :     pandas.DataFrame of shape (n_samples, n_featues).
                Feature matrix.
        """
        raise NotImplementedError()
