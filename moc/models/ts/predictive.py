"""
Predictive objects returned by time-series base models.

Three flavours, matching the three `output_type()`s:

- `PointPredictive`        -- a conditional mean only.
- `QuantilePredictive`     -- a set of conditional quantiles at fixed levels.
- `DistributionPredictive` -- a full conditional density (pdf / cdf / ppf / sampling).

All methods are vectorised over the batch dimension `n`, and accept `y` of shape `(n,)` or
`(n, G)` so that a whole evaluation grid can be scored in one call.
"""

import numpy as np
from scipy import stats


def _as_2d(y, n):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        if y.shape[0] != n:
            raise ValueError(f'expected y of length {n}, got {y.shape}')
        return y[:, None], True
    if y.shape[0] != n:
        raise ValueError(f'expected y with leading dimension {n}, got {y.shape}')
    return y, False


class PointPredictive:
    kind = 'point'

    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=float)

    def __len__(self):
        return len(self.mean)


class QuantilePredictive:
    kind = 'quantile'

    def __init__(self, quantiles, levels):
        self.quantiles = np.asarray(quantiles, dtype=float)   # (n, K)
        self.levels = np.asarray(levels, dtype=float)         # (K,)
        # Quantile crossing is common with independently-fit quantile models; enforce monotonicity.
        self.quantiles = np.sort(self.quantiles, axis=1)

    def __len__(self):
        return len(self.quantiles)

    @property
    def mean(self):
        """Median as the point summary, for scores and metrics that need a centre."""
        return self.quantile_at(0.5)

    def quantile_at(self, level):
        """Linear interpolation across the fitted levels; clamped outside their range."""
        return np.array([
            np.interp(level, self.levels, row) for row in self.quantiles
        ])


class DistributionPredictive:
    """Base class: subclasses implement `logpdf`, `cdf` and `ppf`."""

    kind = 'distribution'

    def __len__(self):
        raise NotImplementedError

    def logpdf(self, y):
        raise NotImplementedError

    def cdf(self, y):
        raise NotImplementedError

    def ppf(self, p):
        raise NotImplementedError

    def pdf(self, y):
        return np.exp(self.logpdf(y))

    @property
    def mean(self):
        return self.ppf(0.5)


class LocationScalePredictive(DistributionPredictive):
    """Normal or standardised Student-t with conditional location and scale."""

    def __init__(self, loc, scale, dist='normal', df=np.inf):
        self.loc = np.asarray(loc, dtype=float)
        self.scale = np.maximum(np.asarray(scale, dtype=float), 1e-12)
        self.dist = dist
        self.df = df

    def __len__(self):
        return len(self.loc)

    def _frozen_std(self):
        if self.dist == 'normal':
            return stats.norm, 1.0
        if self.dist == 't':
            # Standardised so that `scale` is the conditional standard deviation.
            return stats.t(df=self.df), np.sqrt(self.df / (self.df - 2))
        raise ValueError(f'Unsupported distribution {self.dist}')

    def _z(self, y):
        y2, was_1d = _as_2d(y, len(self))
        z = (y2 - self.loc[:, None]) / self.scale[:, None]
        return z, was_1d

    def logpdf(self, y):
        z, was_1d = self._z(y)
        rv, std = self._frozen_std()
        out = rv.logpdf(z * std) + np.log(std) - np.log(self.scale)[:, None]
        return out[:, 0] if was_1d else out

    def cdf(self, y):
        z, was_1d = self._z(y)
        rv, std = self._frozen_std()
        out = rv.cdf(z * std)
        return out[:, 0] if was_1d else out

    def ppf(self, p):
        rv, std = self._frozen_std()
        return self.loc + self.scale * (rv.ppf(p) / std)

    @property
    def mean(self):
        return self.loc


class MixturePredictive(DistributionPredictive):
    """Weighted Gaussian mixture: `sum_j w_j N(y; centre_j, bandwidth^2)` per test point.

    This is the one-dimensional analogue of the repository's existing DRF+KDE model, and the only
    predictive family here that is genuinely multimodal -- which is what makes the density-based
    scores (`nll`, `hpd`) produce unions of intervals rather than plain intervals.

    Components are truncated to the `n_components` largest weights per row and renormalised; the
    full weight vector over the training set would make grid evaluation quadratic in sample size.
    """

    def __init__(self, centres, weights, bandwidth):
        self.centres = np.asarray(centres, dtype=float)        # (n, M)
        w = np.asarray(weights, dtype=float)
        self.weights = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        self.bandwidth = float(max(bandwidth, 1e-8))

    def __len__(self):
        return len(self.centres)

    def logpdf(self, y):
        y2, was_1d = _as_2d(y, len(self))
        # (n, G, M)
        z = (y2[:, :, None] - self.centres[:, None, :]) / self.bandwidth
        log_comp = (
            -0.5 * z ** 2
            - np.log(self.bandwidth)
            - 0.5 * np.log(2 * np.pi)
            + np.log(np.maximum(self.weights, 1e-300))[:, None, :]
        )
        out = _logsumexp(log_comp, axis=-1)
        return out[:, 0] if was_1d else out

    def pdf(self, y):
        """Direct evaluation, bypassing `exp(logpdf)`.

        The log-sum-exp path exists for `logpdf`, where underflow matters. For the density itself
        it costs several extra passes over an (n, G, M) array, and this is the hot loop of the
        whole score pass.
        """
        y2, was_1d = _as_2d(y, len(self))
        z = (y2[:, :, None] - self.centres[:, None, :]) / self.bandwidth
        comp = np.exp(-0.5 * z * z)
        out = np.einsum('ngm,nm->ng', comp, self.weights) / (self.bandwidth * np.sqrt(2 * np.pi))
        return out[:, 0] if was_1d else out

    def cdf(self, y):
        y2, was_1d = _as_2d(y, len(self))
        z = (y2[:, :, None] - self.centres[:, None, :]) / self.bandwidth
        out = np.einsum('ngm,nm->ng', stats.norm.cdf(z), self.weights)
        return out[:, 0] if was_1d else out

    def ppf(self, p):
        """Inverted on a per-row grid; exact to the grid resolution, which is ample for the
        equal-tailed intervals and oracle comparisons that use it."""
        lo = (self.centres - 6 * self.bandwidth).min(axis=1)
        hi = (self.centres + 6 * self.bandwidth).max(axis=1)
        grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, 512)[None, :]
        cdf = self.cdf(grid)
        out = np.empty(len(self))
        for i in range(len(self)):
            out[i] = np.interp(p, cdf[i], grid[i])
        return out

    @property
    def mean(self):
        return (self.centres * self.weights).sum(axis=1)


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))
