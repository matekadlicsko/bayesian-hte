"Scikit-learn based meta-learners."
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from .base import FrequentistMetaLearner


class SLearner(FrequentistMetaLearner):
    """
    Implements of S-learner described in [1]. S-learner estimates conditional average
    treatment effect with the use of a single model.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners
        for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    Parameters
    ----------
    model : sklearn.base.RegressorMixin.
            Base learner.
    """

    def __init__(
        self, model: Optional[RegressorMixin] = None
    ) -> None:
        model = model or LinearRegression()
        super().__init__()
        self.models["model"] = model


    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ):
        X_t = X.assign(treatment=treated)
        self.fit_learner(model=self.models["model"], X=X_t, y=y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        X_control = X.assign(treatment=0)
        X_treated = X.assign(treatment=1)
        m = self.models["model"]
        return pd.Series(m.predict(X_treated) - m.predict(X_control), index=self.index)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        return {"model": self.models["model"].score(X.assign(treatment=treated), y)}


class TLearner(FrequentistMetaLearner):
    """
    Implements of T-learner described in [1]. T-learner fits two separate models to
    estimate conditional average treatment effect.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners
        for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    Parameters
    ----------
    model : sklearn.base.RegressorMixin.
            If specified, it will be used both as treated and untreated model. Either
            model or both of treated_model and untreated_model have to be specified.
    treated_model: sklearn.base.RegressorMixin.
            Model used for predicting target vector for treated values.
    untreated_model: sklearn.base.RegressorMixin.
            Model used for predicting target vector for untreated values.
    """

    def __init__(
        self,
        model: RegressorMixin = LinearRegression(),
        treated_model: Optional[RegressorMixin] = None,
        untreated_model: Optional[RegressorMixin] = None,
    ) -> None:
        super().__init__()
        treated_model = treated_model or deepcopy(model)
        untreated_model = untreated_model or deepcopy(model)

        self.models = {"treated": treated_model, "untreated": untreated_model}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ):
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]
        self.fit_learner(model=self.models["treated"], X=X_t, y=y_t)
        self.fit_learner(model=self.models["untreated"], X=X_u, y=y_u)
        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        return pd.Series(
            treated_model.predict(X) - untreated_model.predict(X), index=self.index
        )

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]
        return {
            "treated": self.models["treated"].score(X_t, y_t),
            "untreated": self.models["untreated"].score(X_u, y_u),
        }


class XLearner(FrequentistMetaLearner):
    """
    Implements of X-learner introduced in [1]. X-learner estimates conditional average
    treatment effect with the use of five separate models.

    [1] Künzel, Sören R., Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners
        for estimating heterogeneous treatment effects using machine learning.
        Proceedings of the national academy of sciences 116, no. 10 (2019): 4156-4165.

    Parameters
    ----------
    model : sklearn.base.RegressorMixin.
            If specified, it will be used in all of the subregressions, except for the
            propensity_score_model. Either model or all of treated_model,
            untreated_model, treated_cate_estimator and untreated_cate_estimator have
            to be specified.
    treated_model : sklearn.base.RegressorMixin.
            Model used for predicting target vector for treated values.
    untreated_model :   sklearn.base.RegressorMixin.
            Model used for predicting target vector for untreated values.
    untreated_cate_estimator :  sklearn.base.RegressorMixin
            Model used for CATE estimation on untreated data.
    treated_cate_estimator :    sklearn.base.RegressorMixin
            Model used for CATE estimation on treated data.
    propensity_score_model :    sklearn.base.ClassifierMixin,
                                default = sklearn.linear_model.LogisticRegression().
            Model used for propensity score estimation.
    """

    def __init__(
        self,
        model: RegressorMixin = LinearRegression(),
        propensity_score_model: ClassifierMixin = LogisticRegression(penalty=None),
        treated_model: Optional[RegressorMixin] = None,
        untreated_model: Optional[RegressorMixin] = None,
        treated_cate_estimator: Optional[RegressorMixin] = None,
        untreated_cate_estimator: Optional[RegressorMixin] = None,
    ):
        super().__init__()
        
        treated_model = treated_model or deepcopy(model)
        untreated_model = untreated_model or deepcopy(model)
        treated_cate_estimator = treated_cate_estimator or deepcopy(model)
        untreated_cate_estimator = untreated_cate_estimator or deepcopy(model)

        self.models = {
            "treated": treated_model,
            "untreated": untreated_model,
            "treated_cate": treated_cate_estimator,
            "untreated_cate": untreated_cate_estimator,
            "propensity": propensity_score_model,
        }

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ):
        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        treated_cate_estimator = self.models["treated_cate"]
        untreated_cate_estimator = self.models["untreated_cate"]
        propensity_score_model = self.models["propensity"]

        # Split data to treated and untreated subsets
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        # Estimate response function
        self.fit_learner(treated_model, X_t, y_t)
        self.fit_learner(untreated_model, X_u, y_u)

        tau_t = y_t - untreated_model.predict(X_t)
        tau_u = treated_model.predict(X_u) - y_u

        # Estimate CATE separately on treated and untreated subsets
        self.fit_learner(treated_cate_estimator, X_t, tau_t)
        self.fit_learner(untreated_cate_estimator, X_u, tau_u)

        # Fit propensity score model
        self.fit_learner(propensity_score_model, X, treated)
        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        cate_estimate_treated = self.models["treated_cate"].predict(X)
        cate_estimate_untreated = self.models["untreated_cate"].predict(X)
        g = self.models["propensity"].predict_proba(X)[:, 1]

        cate = g * cate_estimate_untreated + (1 - g) * cate_estimate_treated
        return pd.Series(cate, index=self.index)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        tau_t = y_t - self.models["untreated"].predict(X_t)
        tau_u = self.models["treated"].predict(X_u) - y_u

        return {
            "treated": self.models["treated"].score(X_t, y_t),
            "untreated": self.models["untreated"].score(X_u, y_u),
            "propensity": self.models["propensity"].score(X, treated),
            "treated_cate": self.models["treated_cate"].score(X_t, tau_t),
            "untreated_cate": self.models["untreated_cate"].score(X_u, tau_u),
        }


class DRLearner(FrequentistMetaLearner):
    """
    Implements of DR-learner also known as doubly robust learner as described in [1].

    [1] Curth, Alicia, Mihaela van der Schaar.
        Nonparametric estimation of heterogeneous treatment effects: From theory to
        learning algorithms. International Conference on Artificial Intelligence and
        Statistics, pp. 1810-1818 (2021).

    Parameters
        ----------
        model : sklearn.base.RegressorMixin.
                If specified, it will be used in all of the subregressions, except for
                the propensity_score_model. Either model or all of treated_model,
                untreated_model, treated_cate_estimator and untreated_cate_estimator
                have to be specified.
        treated_model : sklearn.base.RegressorMixin.
                Model used for predicting target vector for treated values.
        untreated_model :   sklearn.base.RegressorMixin.
                Model used for predicting target vector for untreated values.
        pseudo_outcome_model :  sklearn.base.RegressorMixin
                Model used for pseudo-outcome estimation.
        propensity_score_model :    sklearn.base.ClassifierMixin,
                                   default = sklearn.linear_model.LogisticRegression().
                Model used for propensity score estimation.
        cross_fitting : bool, default=False.
                If True, performs a cross fitting step.
    """

    def __init__(
        self,
        model: RegressorMixin = LinearRegression(),
        propensity_score_model: ClassifierMixin = LogisticRegression(penalty=None),
        treated_model: Optional[RegressorMixin] = None,
        untreated_model: Optional[RegressorMixin] = None,
        pseudo_outcome_model: Optional[RegressorMixin] = None,
        cross_fitting: bool = False,
    ):
        super().__init__()

        self.cross_fitting = cross_fitting

        if model is None and (untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or each of treated_model, untreated_model, \
                treated_cate_estimator, untreated_cate_estimator has to be specified."
            )
        elif not (model is None or untreated_model is None or treated_model is None):
            raise ValueError(
                "Either model or each of treated_model, untreated_model, \
                treated_cate_estimator, untreated_cate_estimator has to be specified."
            )

        if model is not None:
            treated_model = deepcopy(model)
            untreated_model = deepcopy(model)
            pseudo_outcome_model = deepcopy(model)

        # Estimate response function
        self.models = {
            "treated": treated_model,
            "untreated": untreated_model,
            "propensity": propensity_score_model,
            "pseudo_outcome": pseudo_outcome_model,
        }

        cross_fitted_models = {}
        if self.cross_fitting:
            cross_fitted_models = {
                key: deepcopy(value) for key, value in self.models.items()
            }

        self.cross_fitted_models = cross_fitted_models

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ):
        # Split data to two independent samples of equal size
        (X0, X1, y0, y1, treated0, treated1) = train_test_split(
            X, y, treated, stratify=treated, test_size=0.5
        )

        treated_model = self.models["treated"]
        untreated_model = self.models["untreated"]
        propensity_score_model = self.models["propensity"]
        pseudo_outcome_model = self.models["pseudo_outcome"]

        # Second iteration is the cross-fitting step.
        second_iteration = False
        for _ in range(2):
            # Split data to treated and untreated subsets
            X_t, y_t = X0[treated0 == 1], y0[treated0 == 1]
            X_u, y_u = X0[treated0 == 0], y0[treated0 == 0]

            # Estimate response functions
            self.fit_learner(treated_model, X_t, y_t)
            self.fit_learner(untreated_model, X_u, y_u)

            # Fit propensity score model
            self.fit_learner(propensity_score_model, X, treated)

            g = propensity_score_model.predict_proba(X1)[:, 1]
            mu_0 = untreated_model.predict(X1)
            mu_1 = treated_model.predict(X1)
            mu_w = np.where(treated1 == 0, mu_0, mu_1)

            pseudo_outcome = (treated1 - g) / (g * (1 - g)) * (y1 - mu_w) + mu_1 - mu_0

            # Fit pseudo-outcome model
            self.fit_learner(pseudo_outcome_model, X1, pseudo_outcome)

            if self.cross_fitting and not second_iteration:
                # Swap data and estimators
                (X0, X1) = (X1, X0)
                (y0, y1) = (y1, y0)
                (treated0, treated1) = (treated1, treated0)

                treated_model = self.cross_fitted_models["treated"]
                untreated_model = self.cross_fitted_models["untreated"]
                propensity_score_model = self.cross_fitted_models["propensity"]
                pseudo_outcome_model = self.cross_fitted_models["pseudo_outcome"]
                second_iteration = True
            else:
                return self

        return self

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        pred = self.models["pseudo_outcome"].predict(X)

        if self.cross_fitting:
            pred2 = self.cross_fitted_models["pseudo_outcome"].predict(X)
            pred = (pred + pred2) / 2

        return pd.Series(pred, index=self.index)

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treated: pd.Series,
    ) -> Dict[str, np.float64]:
        X_t, y_t = X[treated == 1], y[treated == 1]
        X_u, y_u = X[treated == 0], y[treated == 0]

        g = self.models["propensity"].predict_proba(X)[:, 1]
        mu_0 = self.models["untreated"].predict(X)
        mu_1 = self.models["treated"].predict(X)
        mu_w = np.where(treated == 0, mu_0, mu_1)

        pseudo_outcome = (treated - g) / (g * (1 - g)) * (y - mu_w) + mu_1 - mu_0

        return {
            "treated": self.models["treated"].score(X_t, y_t),
            "untreated": self.models["untreated"].score(X_u, y_u),
            "propensity": self.models["propensity"].score(X, treated),
            "pseudo_outcome": self.models["pseudo_outcome"].score(X, pseudo_outcome),
        }
