"""
Calibration-set schemes: static split, sliding window, and exponentially-weighted (NexCP).

These three form the family that adapts *which calibration data counts*, as opposed to adapting
the target level (`aci.py`) or tracking the quantile directly (`ogd.py`, `pid.py`).
"""

import numpy as np

from .base import (
    HistoryMixin,
    OnlineConformalizer,
    conformal_quantile,
    weighted_conformal_quantile,
)


class Split(OnlineConformalizer):
    """Static split conformal: one threshold from the calibration block, never updated.

    The control condition. It is included precisely because it is expected to degrade, and
    measuring that degradation per dataset is what connects the empirical study to the
    weak-dependence theory.
    """

    name = 'Split'

    def _thresholds_1d(self, s_calib, s_test, alpha):
        return np.full(len(s_test), conformal_quantile(s_calib, alpha))


class Rolling(OnlineConformalizer):
    """Sliding-window conformal: recalibrate on the most recent `window` scores.

    Paired with `EnbPIEnsemble` as the base model, this is EnbPI.
    """

    name = 'Rolling'

    def __init__(self, window=250, **kwargs):
        self.window = window

    def _thresholds_1d(self, s_calib, s_test, alpha):
        n_test = len(s_test)
        full = np.concatenate([s_calib, s_test])
        n_calib = len(s_calib)
        out = np.empty(n_test)
        for t in range(n_test):
            lo = max(0, n_calib + t - self.window)
            out[t] = conformal_quantile(full[lo:n_calib + t], alpha)
        return out

    def describe(self):
        return f'{self.name}(window={self.window})'


class NexCP(OnlineConformalizer, HistoryMixin):
    """Non-exchangeable conformal prediction with geometrically decaying weights.

    `w_i = rho^(age)`, so recent scores dominate. `rho = 1` recovers static split conformal on the
    full history; the HopCPT experiments use `rho = 0.99`.
    """

    name = 'NexCP'

    def __init__(self, rho=0.99, window=2000, **kwargs):
        self.rho = rho
        self.window = window

    def _thresholds_1d(self, s_calib, s_test, alpha):
        n_test = len(s_test)
        full = np.concatenate([s_calib, s_test])
        n_calib = len(s_calib)
        out = np.empty(n_test)
        for t in range(n_test):
            lo = max(0, n_calib + t - self.window)
            hist = full[lo:n_calib + t]
            if len(hist) == 0:
                out[t] = np.inf
                continue
            age = np.arange(len(hist) - 1, -1, -1)     # 0 for the most recent score
            out[t] = weighted_conformal_quantile(hist, self.rho ** age, alpha)
        return out

    def describe(self):
        return f'{self.name}(rho={self.rho})'
