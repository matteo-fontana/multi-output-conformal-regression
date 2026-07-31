"""
SPCI: Sequential Predictive Conformal Inference (Xu & Xie 2023).

The distinguishing idea of this family is that the residual sequence is not noise to be averaged
away but a process with its own structure. SPCI fits a quantile regressor to the recent residual
history and reads `q_t` off its conditional quantile, so serial dependence in the scores becomes a
resource rather than a nuisance.

This is the one scheme here that trains a model inside the online loop, so it declares
`requires_model_refit` and is gated behind `config.ts.allow_sequential_model_methods`. The refit
stride makes the cost tunable; `stride=1` matches the reference at considerable expense.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import OnlineConformalizer


class _QuantileForest:
    """Random forest with leaf-weighted empirical quantiles (Meinshausen's quantile regression
    forest), which is what SPCI uses to model the conditional residual quantile."""

    def __init__(self, n_estimators=50, min_samples_leaf=5, random_state=0):
        self.forest = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=random_state, n_jobs=1,
        )

    def fit(self, x, y):
        self.forest.fit(x, y)
        self.y_train = np.asarray(y, dtype=float)
        self.train_leaves = self.forest.apply(x)
        return self

    def predict_quantile(self, x, level):
        leaves = self.forest.apply(x)
        n_trees = self.train_leaves.shape[1]
        out = np.empty(len(x))
        for i in range(len(x)):
            w = (leaves[i][None, :] == self.train_leaves).sum(axis=1) / n_trees
            total = w.sum()
            if total <= 0:
                out[i] = np.quantile(self.y_train, level)
                continue
            order = np.argsort(self.y_train)
            cum = np.cumsum(w[order]) / total
            idx = np.searchsorted(cum, level, side='left')
            out[i] = self.y_train[order][min(idx, len(order) - 1)]
        return out


class SPCI(OnlineConformalizer):
    """Conditional quantile of the score, regressed on its own recent lags."""

    name = 'SPCI'
    requires_model_refit = True

    def __init__(self, lag=10, window=500, stride=25, n_estimators=50, min_samples_leaf=5,
                 random_state=0, **kwargs):
        self.lag = lag
        self.window = window
        self.stride = stride
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _design(self, series, end):
        """Lagged design matrix over `series[:end]`, restricted to the trailing window."""
        lo = max(self.lag, end - self.window)
        if end - lo < max(2 * self.lag, 20):
            return None, None
        idx = np.arange(lo, end)
        x = np.stack([series[idx - k] for k in range(1, self.lag + 1)], axis=1)
        return x, series[idx]

    def _thresholds_1d(self, s_calib, s_test, alpha):
        from .base import conformal_quantile

        full = np.concatenate([s_calib, s_test])
        n_calib, n_test = len(s_calib), len(s_test)
        out = np.empty(n_test)
        model = None
        fallback = conformal_quantile(s_calib, alpha)

        for t in range(n_test):
            end = n_calib + t
            if t % self.stride == 0 or model is None:
                x, y = self._design(full, end)
                if x is not None:
                    model = _QuantileForest(
                        self.n_estimators, self.min_samples_leaf, self.random_state
                    ).fit(x, y)
            if model is None or end < self.lag:
                out[t] = fallback
                continue
            feat = full[end - np.arange(1, self.lag + 1)][None, :]
            out[t] = model.predict_quantile(feat, 1 - alpha)[0]
        return out

    def describe(self):
        return f'{self.name}(lag={self.lag},stride={self.stride})'
