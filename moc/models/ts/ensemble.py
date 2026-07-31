"""
Bootstrap ensemble with leave-one-out residuals -- the model half of EnbPI.

EnbPI (Xu & Xie 2021) is best understood as two separable ideas:

1. a *model* trick -- fit B bootstrap replicates and, for each training point, aggregate only the
   replicates that did not see it, giving an out-of-sample residual for every training
   observation, so no data has to be held out for calibration; and
2. a *calibration* trick -- take the threshold from a sliding window over the most recent residuals.

Only (1) lives here. (2) is `Rolling` in `moc/conformal/online/`, which every other score can use
too. Pairing this model with `Rolling` reproduces EnbPI; pairing it with `ACI` or `PID` gives
combinations the literature has not reported, which is exactly the point of separating them.
"""

import warnings

import numpy as np

from .models import TSModel, ts_models
from .predictive import PointPredictive


class BootstrapEnsemble(TSModel):
    """Wraps any point model in a bootstrap ensemble that also yields LOO training residuals."""

    provides_loo = True

    def __init__(self, base='Ridge', n_bootstrap=20, aggregation='mean', random_state=0,
                 base_kwargs=None, **kwargs):
        self.base_name = base
        self.n_bootstrap = n_bootstrap
        self.aggregation = aggregation
        self.random_state = random_state
        self.base_kwargs = base_kwargs or {}

    @classmethod
    def output_type(cls):
        return 'point'

    def _agg(self, a, axis):
        if self.aggregation == 'mean':
            return np.nanmean(a, axis=axis)
        if self.aggregation == 'median':
            return np.nanmedian(a, axis=axis)
        raise ValueError(f'Unknown aggregation {self.aggregation}')

    def fit(self, x, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        base_cls = ts_models[self.base_name]
        if base_cls.output_type() != 'point':
            raise ValueError(f'BootstrapEnsemble needs a point base model, got {self.base_name}')

        self.models = []
        in_bag = np.zeros((self.n_bootstrap, n), dtype=bool)
        train_preds = np.full((self.n_bootstrap, n), np.nan)

        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            in_bag[b, idx] = True
            model = base_cls(**self.base_kwargs).fit(x[idx], y[idx])
            self.models.append(model)
            train_preds[b] = model.predict(x).mean

        # Out-of-bag aggregation: for each i, average only over replicates that never saw i.
        oob = np.where(in_bag, np.nan, train_preds)
        with warnings.catch_warnings():
            # A point drawn into every bootstrap sample has an all-NaN column; handled just below.
            warnings.simplefilter('ignore', RuntimeWarning)
            loo_mean = self._agg(oob, axis=0)
        # A point in every bootstrap sample has no OOB prediction; fall back to the full ensemble.
        missing = ~np.isfinite(loo_mean)
        if missing.any():
            loo_mean[missing] = self._agg(train_preds[:, missing], axis=0)

        self.loo_mean = loo_mean
        self.loo_residuals = y - loo_mean
        self.is_fitted = True
        return self

    def predict(self, x):
        preds = np.stack([m.predict(x).mean for m in self.models], axis=0)
        return PointPredictive(self._agg(preds, axis=0))

    def loo_predictive(self):
        """Predictive object over the *training* block, from out-of-bag aggregation."""
        return PointPredictive(self.loo_mean)
