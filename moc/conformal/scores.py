"""
Conformity score families for univariate time series.

This is the axis that the existing time-series benchmarks hold fixed. Separating it from the
calibration scheme is what lets the study ask how much of a method's reported improvement is the
adaptation rule and how much is the score it happens to be paired with.

Interface
---------
A score turns predictions into a stream of nonconformity values, and can also evaluate itself
across an evaluation grid so that prediction sets (possibly non-convex) can be measured exactly.

    prepare(pred_calib, y_calib, pred_test, y_test, alpha, init_scale) -> (s_calib, s_test)
    grid_scores(grid) -> (n_test, n_grid, n_streams)

`n_streams` is 1 for symmetric scores and 2 for the signed residual, where the two tails are
calibrated independently at level `alpha / 2` each. Set membership is the conjunction over streams,
so the interface covers both without special-casing downstream.
"""

import numpy as np


class ScoreFunction:
    name = None
    requires = 'point'          # minimum output_type the base model must provide
    n_streams = 1

    def stream_alphas(self, alpha):
        """How the miscoverage budget is split across independently-calibrated streams."""
        if self.n_streams == 1:
            return np.array([alpha])
        return np.full(self.n_streams, alpha / self.n_streams)

    # -- to implement ------------------------------------------------------------------------

    def _values(self, pred, y, offset):
        raise NotImplementedError

    def _grid(self, pred, grid, offset):
        raise NotImplementedError

    # -- driver ------------------------------------------------------------------------------

    def prepare(self, pred_calib, y_calib, pred_test, y_test, alpha, init_scale=1.0, grid=None):
        self.alpha = alpha
        self.init_scale = init_scale
        self.grid = grid
        self.pred_calib, self.pred_test = pred_calib, pred_test
        self.n_calib, self.n_test = len(y_calib), len(y_test)
        self._setup(pred_calib, y_calib, pred_test, y_test)
        s_calib = np.atleast_2d(self._values(pred_calib, y_calib, offset=0).T).T
        s_test = np.atleast_2d(self._values(pred_test, y_test, offset=self.n_calib).T).T
        return s_calib.reshape(self.n_calib, self.n_streams), s_test.reshape(self.n_test, self.n_streams)

    def _setup(self, pred_calib, y_calib, pred_test, y_test):
        """Hook for scores that need causal auxiliary statistics over the calib+test stream."""
        pass

    def grid_scores(self, grid, chunk=64):
        """Score of every grid point, for every test index. Chunked over the batch dimension
        because the mixture densities build an (n, G, M) intermediate."""
        out = np.empty((self.n_test, len(grid), self.n_streams))
        for lo in range(0, self.n_test, chunk):
            hi = min(lo + chunk, self.n_test)
            sub = _slice_predictive(self.pred_test, lo, hi)
            out[lo:hi] = self._grid(sub, grid, offset=self.n_calib + lo)
        return out


# ------------------------------------------------------------------------------------------------
# Residual-based (point models)
# ------------------------------------------------------------------------------------------------

class AbsResidual(ScoreFunction):
    """|y - ŷ|. The baseline every paper starts from; yields symmetric intervals."""

    name = 'abs_residual'
    requires = 'point'

    def _values(self, pred, y, offset):
        return np.abs(y - pred.mean)[:, None]

    def _grid(self, pred, grid, offset):
        return np.abs(grid[None, :] - pred.mean[:, None])[:, :, None]


class SignedResidual(ScoreFunction):
    """Two one-sided streams, calibrated separately at alpha/2.

    This is the score used in the Conformal PID configs, and the only one here that can produce
    asymmetric intervals -- which matters on skewed series such as electricity prices.
    """

    name = 'signed_residual'
    requires = 'point'
    n_streams = 2

    def _values(self, pred, y, offset):
        r = y - pred.mean
        return np.stack([-r, r], axis=1)

    def _grid(self, pred, grid, offset):
        r = grid[None, :] - pred.mean[:, None]
        return np.stack([-r, r], axis=2)


class NormalizedResidual(ScoreFunction):
    """|y - ŷ| / σ̂_t with a causal σ̂_t from an exponentially-weighted mean of past |residual|.

    σ̂_t uses residuals strictly before t, so the score is computable online. `init_scale` comes
    from the training block, so the first calibration points do not peek at their own residuals.
    """

    name = 'normalized_residual'
    requires = 'point'

    def __init__(self, halflife=25.0, floor=1e-3):
        self.halflife = halflife
        self.floor = floor

    def _setup(self, pred_calib, y_calib, pred_test, y_test):
        resid = np.abs(np.concatenate([
            y_calib - pred_calib.mean, y_test - pred_test.mean,
        ]))
        lam = 0.5 ** (1.0 / self.halflife)
        sigma = np.empty(len(resid))
        state = max(self.init_scale, self.floor)
        for t in range(len(resid)):
            sigma[t] = state                       # depends only on residuals before t
            state = lam * state + (1 - lam) * resid[t]
        self.sigma = np.maximum(sigma, self.floor)

    def _values(self, pred, y, offset):
        s = self.sigma[offset:offset + len(y)]
        return (np.abs(y - pred.mean) / s)[:, None]

    def _grid(self, pred, grid, offset):
        s = self.sigma[offset:offset + len(pred)]
        return (np.abs(grid[None, :] - pred.mean[:, None]) / s[:, None])[:, :, None]


# ------------------------------------------------------------------------------------------------
# Quantile-based (CQR family)
# ------------------------------------------------------------------------------------------------

class CQR(ScoreFunction):
    """max(q̂_lo - y, y - q̂_hi) with q̂ at alpha/2 and 1 - alpha/2 (Romano, Patterson & Candès)."""

    name = 'cqr'
    requires = 'quantile'

    def _bounds(self, pred):
        lo = pred.quantile_at(self.alpha / 2)
        hi = pred.quantile_at(1 - self.alpha / 2)
        return lo, np.maximum(hi, lo)

    def _values(self, pred, y, offset):
        lo, hi = self._bounds(pred)
        return np.maximum(lo - y, y - hi)[:, None]

    def _grid(self, pred, grid, offset):
        lo, hi = self._bounds(pred)
        g = grid[None, :]
        return np.maximum(lo[:, None] - g, g - hi[:, None])[:, :, None]


class CQRr(CQR):
    """CQR normalised by the predicted interval width (Sesia & Candès)."""

    name = 'cqr_r'

    def _scale(self, pred):
        lo, hi = self._bounds(pred)
        return np.maximum(hi - lo, 1e-6)

    def _values(self, pred, y, offset):
        return super()._values(pred, y, offset) / self._scale(pred)[:, None]

    def _grid(self, pred, grid, offset):
        return super()._grid(pred, grid, offset) / self._scale(pred)[:, None, None]


# ------------------------------------------------------------------------------------------------
# Density-based (distribution models). These are the scores that can produce non-convex sets.
# ------------------------------------------------------------------------------------------------

class NLL(ScoreFunction):
    """-log p̂(y | x): the DR-CP score already present in this repository for the multi-output case."""

    name = 'nll'
    requires = 'distribution'

    def _values(self, pred, y, offset):
        return (-pred.logpdf(y))[:, None]

    def _grid(self, pred, grid, offset):
        g = np.broadcast_to(grid[None, :], (len(pred), len(grid)))
        return (-pred.logpdf(g))[:, :, None]


class HPD(ScoreFunction):
    """Highest-predictive-density score: the mass carried by points strictly denser than y.

    In one dimension this is computed exactly by integrating the predictive density over the
    evaluation grid, rather than by Monte Carlo as the multi-output implementation must do.
    """

    name = 'hpd'
    requires = 'distribution'

    def __init__(self, n_grid=1024, span=6.0):
        self.n_grid = n_grid
        self.span = span

    def _setup(self, pred_calib, y_calib, pred_test, y_test):
        # Reuse the evaluation grid as the integration grid when one is available. Beyond halving
        # the number of density evaluations, it guarantees that the score of a realised `y` and the
        # score of a grid point are computed by the same integral -- otherwise coverage measured
        # from the scores and coverage measured from grid membership would disagree slightly.
        if self.grid is not None:
            self.int_grid = np.asarray(self.grid, dtype=float)
        else:
            lo = min(pred_calib.ppf(1e-4).min(), pred_test.ppf(1e-4).min(),
                     y_calib.min(), y_test.min())
            hi = max(pred_calib.ppf(1 - 1e-4).max(), pred_test.ppf(1 - 1e-4).max(),
                     y_calib.max(), y_test.max())
            pad = self.span * 0.05 * (hi - lo)
            self.int_grid = np.linspace(lo - pad, hi + pad, self.n_grid)
        self.n_grid = len(self.int_grid)
        self.dx = self.int_grid[1] - self.int_grid[0]

    def _mass_above(self, pred, dens_at, chunk=64, precomputed=None):
        """For each row, the integrated mass carried by points strictly denser than `dens_at`.

        Fully vectorised: `np.searchsorted` has no batched form, so each row's densities are
        offset into disjoint value ranges and searched in one flat call. The per-row Python loop
        this replaces dominated the runtime of the whole score pass.
        """
        out = np.empty(dens_at.shape)
        n = self.n_grid
        for lo in range(0, len(pred), chunk):
            hi = min(lo + chunk, len(pred))
            b = hi - lo
            if precomputed is not None:
                p = precomputed[lo:hi]
            else:
                sub = _slice_predictive(pred, lo, hi)
                g = np.broadcast_to(self.int_grid[None, :], (b, n))
                p = sub.pdf(g)                                 # (b, n_grid)
            asc = np.sort(p, axis=1)
            cum = np.cumsum(asc[:, ::-1], axis=1) * self.dx    # mass of the k densest points

            d = dens_at[lo:hi]
            span = float(max(asc[:, -1].max(), d.max())) + 1.0
            offsets = (np.arange(b) * span)[:, None]
            flat = (asc + offsets).ravel()                     # globally sorted by construction
            idx = np.searchsorted(flat, (d + offsets).ravel(), side='right')
            idx = idx.reshape(b, -1) - (np.arange(b) * n)[:, None]
            count = n - np.clip(idx, 0, n)                     # points strictly denser than d
            mass = np.take_along_axis(cum, np.clip(count - 1, 0, n - 1), axis=1)
            out[lo:hi] = np.where(count == 0, 0.0, mass)
        return np.clip(out, 0.0, 1.0)

    def _values(self, pred, y, offset):
        dens = pred.pdf(y)[:, None]
        return self._mass_above(pred, dens)

    def _grid(self, pred, grid, offset):
        g = np.broadcast_to(grid[None, :], (len(pred), len(grid)))
        dens = pred.pdf(g)
        # When the evaluation grid is the integration grid, the density we just computed *is* the
        # one the integral needs, so it is threaded through instead of being recomputed.
        same = len(grid) == self.n_grid and np.array_equal(grid, self.int_grid)
        return self._mass_above(pred, dens, precomputed=dens if same else None)[:, :, None]


class PIT(ScoreFunction):
    """|F̂(y | x) - 1/2|: equal-tailed by construction, and the natural rank-based score."""

    name = 'pit'
    requires = 'distribution'

    def _values(self, pred, y, offset):
        return np.abs(pred.cdf(y) - 0.5)[:, None]

    def _grid(self, pred, grid, offset):
        g = np.broadcast_to(grid[None, :], (len(pred), len(grid)))
        return np.abs(pred.cdf(g) - 0.5)[:, :, None]


# ------------------------------------------------------------------------------------------------

def _slice_predictive(pred, lo, hi):
    """Row-slice a predictive object without knowing its concrete class."""
    from moc.models.ts.predictive import (
        LocationScalePredictive, MixturePredictive, PointPredictive, QuantilePredictive,
    )

    if isinstance(pred, PointPredictive):
        return PointPredictive(pred.mean[lo:hi])
    if isinstance(pred, QuantilePredictive):
        return QuantilePredictive(pred.quantiles[lo:hi], pred.levels)
    if isinstance(pred, LocationScalePredictive):
        return LocationScalePredictive(pred.loc[lo:hi], pred.scale[lo:hi], pred.dist, pred.df)
    if isinstance(pred, MixturePredictive):
        return MixturePredictive(pred.centres[lo:hi], pred.weights[lo:hi], pred.bandwidth)
    raise TypeError(f'Cannot slice predictive of type {type(pred)}')


scores = {
    'abs_residual': AbsResidual,
    'signed_residual': SignedResidual,
    'normalized_residual': NormalizedResidual,
    'cqr': CQR,
    'cqr_r': CQRr,
    'nll': NLL,
    'hpd': HPD,
    'pit': PIT,
}

# Which scores each base-model output type can support.
SCORES_BY_OUTPUT_TYPE = {
    'point': ['abs_residual', 'signed_residual', 'normalized_residual'],
    'quantile': ['abs_residual', 'signed_residual', 'normalized_residual', 'cqr', 'cqr_r'],
    'distribution': ['abs_residual', 'signed_residual', 'normalized_residual', 'nll', 'hpd', 'pit'],
}


def compatible(score_name, output_type):
    return score_name in SCORES_BY_OUTPUT_TYPE[output_type]
