"""
The ACI family: schemes that adapt the *target level* alpha_t rather than the threshold directly.

`ACI` is Gibbs & Candès (2021); `DtACI` is the dynamically-tuned variant of Gibbs & Candès (2024),
whose expert-aggregation constants follow the reference implementation in
`salesforce/online_conformal` (`faci.py`).

Both share the same alpha recursion, kept in one place so that a comparison between them isolates
the aggregation rule rather than incidental implementation differences.
"""

import numpy as np
from scipy.special import logsumexp

from .base import HistoryMixin, OnlineConformalizer, conformal_quantile, pinball_loss


def alpha_step(alpha_t, alpha_target, gamma, err):
    """The ACI update: `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`."""
    return alpha_t + gamma * (alpha_target - err)


class ACI(OnlineConformalizer, HistoryMixin):
    """Adaptive Conformal Inference (Gibbs & Candès 2021).

    `alpha_t` drifts up when recent intervals over-cover and down when they under-cover. When
    `alpha_t` leaves `[0, 1]` the prescription is an infinite (resp. empty) set; those steps are
    counted so that the degenerate-set rate can be reported rather than silently distorting the
    mean width (guard 6 of docs §8).
    """

    name = 'ACI'

    def __init__(self, gamma=0.01, window=None, **kwargs):
        self.gamma = gamma
        self.window = window
        self.n_infinite = 0
        self.n_empty = 0

    def _thresholds_1d(self, s_calib, s_test, alpha):
        n_test = len(s_test)
        out = np.empty(n_test)
        alpha_t = alpha
        full, n_calib = self.full_stream(s_calib, s_test)
        for t in range(n_test):
            hist = self.history_view(full, n_calib, t, self.window)
            if alpha_t <= 0:
                q = np.inf
                self.n_infinite += 1
            elif alpha_t >= 1:
                q = -np.inf
                self.n_empty += 1
            else:
                q = conformal_quantile(hist, alpha_t)
            out[t] = q
            err = float(s_test[t] > q)
            alpha_t = alpha_step(alpha_t, alpha, self.gamma, err)
        return out

    def describe(self):
        return f'{self.name}(gamma={self.gamma})'


class DtACI(OnlineConformalizer, HistoryMixin):
    """Dynamically-tuned ACI / FACI (Gibbs & Candès 2024).

    Runs one ACI expert per candidate learning rate and aggregates their `alpha_t` by exponential
    weighting on the pinball loss, removing the learning-rate choice that ACI leaves to the user.
    The `eta`/`sigma` heuristics and the `beta = mean(history >= s)` loss target follow the
    reference implementation.
    """

    name = 'DtACI'

    def __init__(self, gammas=None, interval=100, window=None, **kwargs):
        self.gammas = np.asarray(gammas if gammas is not None
                                 else [0.001 * 2 ** k for k in range(8)], dtype=float)
        self.interval = interval           # `I` in the reference: the horizon the constants target
        self.window = window

    def _eta(self, alpha):
        k = len(self.gammas)
        denom = ((1 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1 - alpha) ** 3) / 3
        return np.sqrt(3 / self.interval) * np.sqrt((np.log(self.interval * k) + 2) / denom)

    def _thresholds_1d(self, s_calib, s_test, alpha):
        k = len(self.gammas)
        alphas = np.full(k, alpha, dtype=float)
        log_w = np.zeros(k)
        eta = self._eta(alpha)
        sigma = 1.0 / (2 * self.interval)

        out = np.empty(len(s_test))
        full, n_calib = self.full_stream(s_calib, s_test)
        for t in range(len(s_test)):
            hist = self.history_view(full, n_calib, t, self.window)
            p = np.exp(log_w - logsumexp(log_w))
            alpha_bar = float(np.dot(p, alphas))
            out[t] = conformal_quantile(hist, np.clip(alpha_bar, 0.0, 1.0))

            s = s_test[t]
            # `beta` is the level at which the realised score sits in the current history: the
            # aggregation is over levels, which is what makes the experts comparable.
            beta = float(np.mean(hist >= s)) if len(hist) else alpha
            # The pinball level is `alpha`, not `1 - alpha`: the experts are compared on the
            # *level* scale, matching `pinball_loss(beta, alphas, 1 - coverage)` in the reference.
            losses = pinball_loss(beta, alphas, alpha)
            wbar = log_w - eta * losses
            log_w = logsumexp(
                np.stack([wbar, np.full(k, logsumexp(wbar))]),
                b=np.stack([np.full(k, 1 - sigma), np.full(k, sigma / k)]),
                axis=0,
            )
            log_w -= logsumexp(log_w)
            err = (alphas > beta).astype(float)
            alphas = np.clip(alphas + self.gammas * (alpha - err), 0.0, 1.0)
        return out

    def describe(self):
        return f'{self.name}(k={len(self.gammas)})'
