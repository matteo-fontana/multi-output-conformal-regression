"""
Conformal PID control (Angelopoulos, Candès & Tibshirani 2023).

The paper's reading of the field: ACI is pure integral control on the coverage error, so the
natural completions are a proportional term (online quantile tracking) and a predictive term (a
*scorecaster* that forecasts the score itself). The three variants here isolate those terms:

    QuantileTracker  P only
    PID              P + I, with the log saturation function
    PIDScorecaster   P + I + a Theta-model forecast of the score

Follows `aangelopoulos/conformal-time-series/core/methods.py`
(`quantile_integrator_log_scorecaster`), specialised to one-step-ahead prediction.
"""

import logging

import numpy as np

from .base import OnlineConformalizer

log = logging.getLogger('moc')


def _mytan(x):
    if x >= np.pi / 2:
        return np.inf
    if x <= -np.pi / 2:
        return -np.inf
    return np.tan(x)


def saturation_fn_log(x, t, Csat, KI):
    """Integral term with a saturating nonlinearity, so a long coverage debt cannot blow up `q`."""
    if KI == 0:
        return 0.0
    return KI * _mytan(x * np.log(t + 1) / (Csat * (t + 1)))


class PID(OnlineConformalizer):
    """Proportional + saturated-integral quantile tracking."""

    name = 'PID'
    scorecast = False

    def __init__(self, lr=0.1, Csat=5.0, KI=10.0, T_burnin=None, proportional_lr=True,
                 scorecast_stride=50, seasonal_period=1, **kwargs):
        self.lr = lr
        self.Csat = Csat
        self.KI = KI
        self.T_burnin = T_burnin
        self.proportional_lr = proportional_lr
        self.scorecast_stride = scorecast_stride
        self.seasonal_period = seasonal_period

    def _thresholds_1d(self, s_calib, s_test, alpha):
        full = np.concatenate([s_calib, s_test])
        n_calib, n = len(s_calib), len(full)
        burnin = self.T_burnin if self.T_burnin is not None else max(1, min(n_calib, 500))

        qs = np.zeros(n)
        qts = np.zeros(n)
        covered = np.zeros(n)
        scorecasts = np.zeros(n)

        # Start the tracker at the split-conformal threshold rather than zero: without a warm start
        # the proportional term spends the first few hundred steps climbing out of q = 0.
        if n_calib > 0:
            from .base import conformal_quantile
            q0 = conformal_quantile(s_calib, alpha)
            if np.isfinite(q0):
                qs[0] = qts[0] = q0

        next_scorecast_fit = 0
        model_forecast = 0.0

        for t in range(n):
            lo = max(t - burnin, 0)
            if self.proportional_lr and t > 0:
                window = full[lo:t]
                lr_t = self.lr * (window.max() - window.min())
            else:
                lr_t = self.lr

            covered[t] = qs[t] >= full[t]
            grad = alpha if covered[t] else -(1 - alpha)
            integrator_arg = (1 - covered)[:t].sum() - t * alpha
            integrator = saturation_fn_log(integrator_arg, t, self.Csat, self.KI)

            if self.scorecast and t >= next_scorecast_fit and t > burnin and t < n - 1:
                model_forecast = self._scorecast(full[:t])
                next_scorecast_fit = t + self.scorecast_stride
            if self.scorecast and t < n - 1:
                scorecasts[t + 1] = model_forecast

            if t < n - 1:
                qts[t + 1] = qts[t] - lr_t * grad
                qs[t + 1] = qts[t + 1] + integrator + scorecasts[t + 1]

        return qs[n_calib:]

    def _scorecast(self, past_scores):
        return 0.0

    def describe(self):
        return f'{self.name}(lr={self.lr},KI={self.KI})'


class QuantileTracker(PID):
    """Proportional control only: online quantile regression on the score stream."""

    name = 'QuantileTracker'

    def __init__(self, lr=0.1, **kwargs):
        super().__init__(lr=lr, Csat=1.0, KI=0.0, **kwargs)

    def describe(self):
        return f'{self.name}(lr={self.lr})'


class PIDScorecaster(PID):
    """PID plus a Theta-model forecast of the score, refit every `scorecast_stride` steps.

    Refitting at every step as the reference does is affordable for a single published
    configuration but not for a full factorial sweep, so the stride is exposed; `stride=1`
    reproduces the reference exactly.
    """

    name = 'PID+Scorecaster'
    scorecast = True
    expensive = True

    def _scorecast(self, past_scores):
        from statsmodels.tsa.forecasting.theta import ThetaModel

        curr = np.nan_to_num(np.asarray(past_scores, dtype=float))
        period = max(int(self.seasonal_period), 1)
        if len(curr) < max(2 * period, 10):
            return 0.0
        try:
            model = ThetaModel(curr, period=period if period > 1 else None).fit()
            return float(np.asarray(model.forecast(1))[0])
        except Exception as e:                       # statsmodels is brittle on degenerate windows
            log.debug(f'scorecaster failed at length {len(curr)}: {e}')
            return 0.0
