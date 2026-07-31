"""
Base classes and quantile primitives for online conformal calibration.

The central design point (see docs/TIMESERIES_TESTBED_PLAN.md §3.2): almost every method in this
literature updates a *threshold*, not a model. So the expensive, model-bound work -- computing the
score stream -- happens once, vectorised, and everything here is an O(T) numpy recursion over that
stream. Only `SPCI` (and rolling-refit configurations) break the pattern, and they say so via
`requires_model_refit`.

Every scheme is asked for the same thing: given the calibration scores and the test scores,
produce the threshold `q_t` that would have been used at each test step, using only information
available strictly before `t`.
"""

import numpy as np


# ------------------------------------------------------------------------------------------------
# Quantile primitives
# ------------------------------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    """The standard split-conformal quantile: the ⌈(1-α)(n+1)⌉-th smallest score, or +∞.

    Mirrors `moc/conformal/base_conformalizer.py::conformal_quantile` for the 1-D numpy case.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.inf
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    k = int(np.ceil((1 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    return np.partition(scores, k - 1)[k - 1]


def weighted_conformal_quantile(scores, weights, alpha):
    """Weighted quantile of Barber, Candès, Ramdas & Tibshirani (2023).

    Normalisation is `w̃_i = w_i / (Σ_j w_j + 1)` with the remaining `1 / (Σ_j w_j + 1)` mass on an
    atom at `+∞`, which is what makes the procedure conservative rather than merely heuristic.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(scores) == 0:
        return np.inf
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    p = w / (w.sum() + 1.0)
    cum = np.cumsum(p)
    idx = np.searchsorted(cum, 1 - alpha, side='left')
    if idx >= len(s):
        return np.inf                      # the +∞ atom carries the remaining mass
    return s[idx]


def pinball_loss(y, q, level):
    return np.maximum(level * (y - q), (1 - level) * (q - y))


def pinball_loss_grad(y, q, level):
    """d/dq of the pinball loss at level `level`."""
    return -level * (y > q) + (1 - level) * (y < q)


# ------------------------------------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------------------------------------

class OnlineConformalizer:
    """Maps a score stream to a threshold stream.

    Subclasses implement `_thresholds_1d`. Multi-stream scores (the signed residual) are handled
    by running the scheme independently per stream at the per-stream miscoverage level.
    """

    name = None
    requires_model_refit = False

    def thresholds(self, s_calib, s_test, alphas):
        s_calib = np.atleast_2d(s_calib.T).T
        s_test = np.atleast_2d(s_test.T).T
        n_streams = s_calib.shape[1]
        alphas = np.broadcast_to(np.asarray(alphas, dtype=float), (n_streams,))
        out = np.empty((s_test.shape[0], n_streams))
        for k in range(n_streams):
            out[:, k] = self._thresholds_1d(s_calib[:, k], s_test[:, k], float(alphas[k]))
        return out

    def _thresholds_1d(self, s_calib, s_test, alpha):
        raise NotImplementedError

    def describe(self):
        return self.name


class HistoryMixin:
    """Helper for schemes whose threshold is a quantile over a growing or sliding history."""

    @staticmethod
    def history_view(s_calib, s_test, t, window):
        """Scores observable strictly before test index `t`."""
        if t == 0:
            hist = s_calib
        else:
            hist = np.concatenate([s_calib, s_test[:t]])
        if window is not None and window > 0 and len(hist) > window:
            hist = hist[-window:]
        return hist
