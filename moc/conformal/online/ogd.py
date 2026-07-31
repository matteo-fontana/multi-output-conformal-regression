"""
Online-learning schemes that track the score quantile directly.

`SFOGD` is scale-free online gradient descent on the pinball loss (Orabona & Pál 2016, applied to
conformal calibration by Bhatnagar et al. 2023). `SAOCP` layers Coin Betting for Changing
Environments over a set of SF-OGD learners with geometrically-covering lifetimes, giving
*strongly adaptive* regret -- good coverage on every sub-interval, not just on average.

Both follow the reference implementation in `salesforce/online_conformal` (`ogd.py`, `saocp.py`).
The one deliberate deviation: the reference clamps the threshold at zero because it always works
with `|residual|`, whereas scores here may legitimately be negative (signed residual, CQR). The
clamp is therefore applied only when the calibration scores are non-negative.
"""

import numpy as np

from .base import OnlineConformalizer, pinball_loss, pinball_loss_grad


def _default_scale(s_calib):
    if len(s_calib) == 0:
        return 1.0
    scale = float(np.max(np.abs(s_calib)) * np.sqrt(3))
    return scale if scale > 0 else 1.0


class SFOGD(OnlineConformalizer):
    """Scale-free online gradient descent on the pinball loss."""

    name = 'SF-OGD'

    def __init__(self, scale=None, **kwargs):
        self.scale = scale

    def _thresholds_1d(self, s_calib, s_test, alpha):
        level = 1 - alpha
        scale = self.scale if self.scale is not None else _default_scale(s_calib)
        clip_min = 0.0 if (len(s_calib) and s_calib.min() >= 0) else -np.inf

        q, grad_norm = 0.0, 0.0

        def step(s, q, grad_norm):
            grad = pinball_loss_grad(s, q, level)
            grad_norm += grad ** 2
            if grad_norm > 0:
                q = max(clip_min, q - scale / np.sqrt(3 * grad_norm) * grad)
            return q, grad_norm

        # Warm up on the calibration block, exactly as the reference does.
        for s in s_calib:
            q, grad_norm = step(s, q, grad_norm)

        out = np.empty(len(s_test))
        for t, s in enumerate(s_test):
            out[t] = q
            q, grad_norm = step(s, q, grad_norm)
        return out


class _CBExpert:
    """One SF-OGD learner with a finite lifetime and a coin-betting weight."""

    def __init__(self, birth, scale, alpha, q0, lifetime_base=8, clip_min=-np.inf):
        self.scale = scale
        self.base_lr = scale / np.sqrt(3)
        self.alpha = alpha
        self.q = q0
        self.grad_norm = 0.0
        self.clip_min = clip_min

        # Lifetime from the geometric covering of Hazan & Seshadhri (2007), as in the reference.
        t, u = birth, 0
        while t % 2 == 0:
            t //= 2
            u += 1
        self.lifetime = lifetime_base * 2 ** u

        self.z = 0.0        # cumulative advantage over the meta-prediction
        self.wz = 0.0
        self.age = 0

    @property
    def expired(self):
        return self.age > self.lifetime

    @property
    def w(self):
        return 0.0 if self.age == 0 else self.z / self.age * (1 + self.wz)

    def loss(self, s):
        return pinball_loss(s, self.q, 1 - self.alpha)

    def update(self, s, meta_loss):
        w = self.w
        denom = self.scale * max(self.alpha, 1 - self.alpha)
        g = np.clip((meta_loss - self.loss(s)) / denom, -1.0 * (w > 0), 1.0)
        self.z += g
        self.wz += g * w
        self.age += 1

        grad = pinball_loss_grad(s, self.q, 1 - self.alpha)
        self.grad_norm += grad ** 2
        if self.grad_norm > 0:
            self.q = max(self.clip_min, self.q - self.base_lr / np.sqrt(self.grad_norm) * grad)


class SAOCP(OnlineConformalizer):
    """Strongly Adaptive Online Conformal Prediction (Bhatnagar et al. 2023)."""

    name = 'SAOCP'

    def __init__(self, lifetime=8, scale=None, **kwargs):
        self.lifetime = lifetime
        self.scale = scale

    def _thresholds_1d(self, s_calib, s_test, alpha):
        scale = self.scale if self.scale is not None else _default_scale(s_calib)
        clip_min = 0.0 if (len(s_calib) and s_calib.min() >= 0) else -np.inf
        experts = {}
        clock = [1]

        def prior(keys):
            p = np.array([1.0 / (t ** 2 * (1 + np.floor(np.log2(t)))) for t in keys])
            return p / p.sum()

        def aggregate():
            if not experts:
                return 0.0
            keys = list(experts)
            pri = prior(keys)
            w = np.array([max(0.0, experts[t].w) for t in keys])
            p = pri * w
            p = p / p.sum() if p.sum() > 0 else pri
            return float(np.dot(p, [experts[t].q for t in keys]))

        def step(s):
            q_prev = aggregate()
            for t in [t for t, e in experts.items() if e.expired]:
                experts.pop(t)
            experts[clock[0]] = _CBExpert(clock[0], scale, alpha, q_prev,
                                          lifetime_base=self.lifetime, clip_min=clip_min)
            q = aggregate()
            meta_loss = pinball_loss(s, q, 1 - alpha)
            for e in experts.values():
                e.update(s, meta_loss)
            clock[0] += 1
            return q

        for s in s_calib:
            step(s)

        out = np.empty(len(s_test))
        for t, s in enumerate(s_test):
            out[t] = step(s)
        return out
