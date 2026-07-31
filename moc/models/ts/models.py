"""
Base predictors for the time-series testbed.

Deliberately spans the cost/quality range a review should cover, from a seasonal-naive floor to a
multimodal conditional density. The floor matters: it isolates how much of a conformal method's
coverage comes from calibration rather than from the model.

All models share a small interface:

    output_type() -> 'point' | 'quantile' | 'distribution'
    fit(x, y)
    predict(x)   -> a Predictive object from moc.models.ts.predictive
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .predictive import (
    DistributionPredictive,
    LocationScalePredictive,
    MixturePredictive,
    PointPredictive,
    QuantilePredictive,
)

log = logging.getLogger('moc')

DEFAULT_QUANTILE_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class TSModel:
    is_fitted = False

    @classmethod
    def output_type(cls):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------
# Point predictors
# ------------------------------------------------------------------------------------------------

class Naive(TSModel):
    """ŷ_t = y_{t-1}. The cost floor."""

    def __init__(self, lag_index=0, **kwargs):
        self.lag_index = lag_index

    @classmethod
    def output_type(cls):
        return 'point'

    def fit(self, x, y):
        self.is_fitted = True
        return self

    def predict(self, x):
        return PointPredictive(x[:, self.lag_index])


class SeasonalNaive(Naive):
    """ŷ_t = y_{t-s}, using the seasonal lag column when the datamodule created one."""

    def __init__(self, seasonal_index=None, **kwargs):
        self.seasonal_index = seasonal_index
        super().__init__(lag_index=0)

    def fit(self, x, y):
        # The datamodule appends the seasonal lag directly after the plain lags when it exists.
        self.lag_index = self.seasonal_index if self.seasonal_index is not None else 0
        self.is_fitted = True
        return self


class RidgeModel(TSModel):
    """Linear autoregression with L2 regularisation."""

    def __init__(self, alpha=1.0, **kwargs):
        self.model = Ridge(alpha=alpha)

    @classmethod
    def output_type(cls):
        return 'point'

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, x):
        return PointPredictive(self.model.predict(x))


class LGBMModel(TSModel):
    """Gradient boosting on lag features: the workhorse of the forecasting-competition literature."""

    def __init__(self, n_estimators=200, learning_rate=0.05, num_leaves=31, **kwargs):
        self.kwargs = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, verbose=-1,
        )

    @classmethod
    def output_type(cls):
        return 'point'

    def fit(self, x, y):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(**self.kwargs)
        self.model.fit(x, y)
        self.is_fitted = True
        return self

    def predict(self, x):
        return PointPredictive(self.model.predict(x))


# ------------------------------------------------------------------------------------------------
# Quantile predictors (feed the CQR score family)
# ------------------------------------------------------------------------------------------------

class LGBMQuantile(TSModel):
    """One gradient-boosting model per quantile level, with the pinball objective."""

    def __init__(self, levels=DEFAULT_QUANTILE_LEVELS, n_estimators=200, learning_rate=0.05,
                 num_leaves=31, **kwargs):
        self.levels = tuple(levels)
        self.kwargs = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, verbose=-1,
        )

    @classmethod
    def output_type(cls):
        return 'quantile'

    def fit(self, x, y):
        import lightgbm as lgb
        self.models = []
        for level in self.levels:
            m = lgb.LGBMRegressor(objective='quantile', alpha=level, **self.kwargs)
            m.fit(x, y)
            self.models.append(m)
        self.is_fitted = True
        return self

    def predict(self, x):
        q = np.stack([m.predict(x) for m in self.models], axis=1)
        return QuantilePredictive(q, self.levels)


class LinearQuantile(TSModel):
    """Linear quantile regression, fit by gradient descent on the pinball loss.

    Cheap, deterministic, and a useful contrast with LGBMQuantile: it cannot represent
    heteroscedasticity beyond what is linear in the features.
    """

    def __init__(self, levels=DEFAULT_QUANTILE_LEVELS, n_iter=500, lr=0.05, l2=1e-4, **kwargs):
        self.levels = tuple(levels)
        self.n_iter, self.lr, self.l2 = n_iter, lr, l2

    @classmethod
    def output_type(cls):
        return 'quantile'

    def fit(self, x, y):
        n, p = x.shape
        xb = np.concatenate([np.ones((n, 1)), x], axis=1)
        self.coef = np.zeros((len(self.levels), p + 1))
        for k, level in enumerate(self.levels):
            w = np.zeros(p + 1)
            w[0] = np.quantile(y, level)
            for _ in range(self.n_iter):
                resid = y - xb @ w
                grad = -xb.T @ (np.where(resid >= 0, level, level - 1.0)) / n + self.l2 * w
                w = w - self.lr * grad
            self.coef[k] = w
        self.is_fitted = True
        return self

    def predict(self, x):
        xb = np.concatenate([np.ones((len(x), 1)), x], axis=1)
        return QuantilePredictive(xb @ self.coef.T, self.levels)


# ------------------------------------------------------------------------------------------------
# Distribution predictors (feed nll / hpd / pit)
# ------------------------------------------------------------------------------------------------

class GaussianRidge(TSModel):
    """Ridge conditional mean plus a conditional scale fitted to the log squared residuals.

    Unimodal by construction, so `hpd` and `pit` coincide here; its role is to be the parametric
    contrast to the mixture model below.
    """

    def __init__(self, alpha=1.0, **kwargs):
        self.mean_model = Ridge(alpha=alpha)
        self.scale_model = Ridge(alpha=alpha)

    @classmethod
    def output_type(cls):
        return 'distribution'

    def fit(self, x, y):
        self.mean_model.fit(x, y)
        resid = y - self.mean_model.predict(x)
        self.scale_model.fit(x, np.log(np.maximum(resid ** 2, 1e-8)))
        self.is_fitted = True
        return self

    def predict(self, x):
        loc = self.mean_model.predict(x)
        scale = np.sqrt(np.exp(np.clip(self.scale_model.predict(x), -20, 20)))
        return LocationScalePredictive(loc, scale)


class RFKDE(TSModel):
    """Random-forest weights plus a Gaussian kernel: the 1-D analogue of the repo's DRF+KDE.

    For a test point, the forest induces a weight over training observations (the fraction of trees
    in which the training point shares a leaf). Smoothing those weighted atoms gives a genuinely
    multimodal conditional density, which is what makes `hpd` produce non-convex prediction sets.
    """

    def __init__(self, n_estimators=100, min_samples_leaf=10, n_components=40,
                 bandwidth=None, random_state=0, **kwargs):
        self.forest = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=1,
        )
        self.n_components = n_components
        self.bandwidth = bandwidth

    @classmethod
    def output_type(cls):
        return 'distribution'

    def fit(self, x, y):
        self.forest.fit(x, y)
        self.y_train = np.asarray(y, dtype=float)
        self.train_leaves = self.forest.apply(x)          # (n_train, n_trees)
        if self.bandwidth is None:
            # Silverman's rule on the training targets, floored away from zero.
            n = len(y)
            self.bandwidth = max(
                0.9 * min(self.y_train.std(), _iqr(self.y_train) / 1.349) * n ** (-1 / 5), 1e-3
            )
        self.is_fitted = True
        return self

    def predict(self, x, chunk=256):
        test_leaves = self.forest.apply(x)
        n_trees = self.train_leaves.shape[1]
        centres = np.empty((len(x), self.n_components))
        weights = np.empty((len(x), self.n_components))
        for lo in range(0, len(x), chunk):
            hi = min(lo + chunk, len(x))
            # (b, n_train): how often each training point lands in the same leaf.
            match = (test_leaves[lo:hi, None, :] == self.train_leaves[None, :, :]).sum(axis=2)
            w = match / n_trees
            m = min(self.n_components, w.shape[1])
            idx = np.argpartition(-w, m - 1, axis=1)[:, :m]
            rows = np.arange(hi - lo)[:, None]
            centres[lo:hi, :m] = self.y_train[idx]
            weights[lo:hi, :m] = w[rows, idx]
            if m < self.n_components:                      # tiny training sets
                centres[lo:hi, m:] = self.y_train.mean()
                weights[lo:hi, m:] = 0.0
        # A degenerate row (no shared leaves at all) falls back to the marginal.
        degenerate = weights.sum(axis=1) <= 0
        if degenerate.any():
            weights[degenerate] = 1.0 / self.n_components
        return MixturePredictive(centres, weights, self.bandwidth)


class OracleModel(TSModel):
    """Exact conditional law. Synthetic datasets only; the upper bound on achievable performance."""

    def __init__(self, datamodule=None, **kwargs):
        self.datamodule = datamodule

    @classmethod
    def output_type(cls):
        return 'distribution'

    def fit(self, x, y):
        self.is_fitted = True
        return self

    def predict(self, x, split=None):
        if split is None or split.oracle_loc is None:
            raise ValueError('OracleModel requires a split carrying the oracle conditional law')
        return LocationScalePredictive(
            split.oracle_loc, split.oracle_scale,
            dist=self.datamodule.oracle_dist, df=self.datamodule.oracle_df,
        )

    @property
    def needs_split(self):
        return True


def _iqr(a):
    q75, q25 = np.percentile(a, [75, 25])
    return max(q75 - q25, 1e-8)


ts_models = {
    'Naive': Naive,
    'SeasonalNaive': SeasonalNaive,
    'Ridge': RidgeModel,
    'LGBM': LGBMModel,
    'LGBMQuantile': LGBMQuantile,
    'LinearQuantile': LinearQuantile,
    'GaussianRidge': GaussianRidge,
    'RFKDE': RFKDE,
    'Oracle': OracleModel,
}
