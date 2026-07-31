"""
Synthetic univariate time-series generators with a known conditional law.

Every generator returns, alongside the series, the exact conditional distribution of `y_t` given
the observed past. That is what makes the synthetic group the only place where conditional coverage
can be measured rather than proxied, and where the weak-dependence bounds of Stocker et al. (2025)
can be checked against a known mixing structure.

Each generator returns a `SyntheticSeries` whose `loc`/`scale`/`dist` describe
    y_t | F_{t-1}  ~  loc_t + scale_t * D
with D standard normal or standard Student-t with `df` degrees of freedom.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class SyntheticSeries:
    y: np.ndarray                     # (T,)
    loc: np.ndarray                   # (T,) conditional location given the past
    scale: np.ndarray                 # (T,) conditional scale given the past
    dist: str = 'normal'              # 'normal' | 't'
    df: float = np.inf
    exog: np.ndarray = None           # (T, k) or None
    exog_names: list = field(default_factory=list)
    changepoints: np.ndarray = None   # indices of known regime changes, or None
    # Theoretical mixing decay rate where it is known in closed form (see docs §7.2).
    mixing_rate: float = None

    def conditional_quantile(self, p):
        """Exact conditional quantile of y_t given the past, for each t. `p` is a scalar."""
        if self.dist == 'normal':
            z = stats.norm.ppf(p)
        elif self.dist == 't':
            # Standardised Student-t so that `scale` is the conditional standard deviation.
            z = stats.t.ppf(p, df=self.df) / np.sqrt(self.df / (self.df - 2))
        else:
            raise ValueError(f'Unsupported conditional distribution: {self.dist}')
        return self.loc + self.scale * z

    def oracle_interval(self, alpha):
        """Exact equal-tailed oracle interval at level 1 - alpha."""
        return self.conditional_quantile(alpha / 2), self.conditional_quantile(1 - alpha / 2)


def _rng(seed):
    return np.random.default_rng(seed)


def ar1_gauss(T, seed, phi=0.6, sigma=1.0):
    """AR(1) with homoscedastic Gaussian noise. The control: nothing to adapt to."""
    rng = _rng(seed)
    eps = rng.normal(0, sigma, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + eps[t]
    loc = np.concatenate([[0.0], phi * y[:-1]])
    return SyntheticSeries(y=y, loc=loc, scale=np.full(T, sigma), mixing_rate=abs(phi))


def ar1_garch(T, seed, phi=0.5, omega=0.05, a=0.1, b=0.85):
    """AR(1) mean with GARCH(1,1) conditional variance.

    The conditional scale varies by an order of magnitude, which is what separates
    volatility-normalised scores from plain absolute residuals.
    """
    rng = _rng(seed)
    z = rng.normal(0, 1, T)
    y = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / max(1e-8, 1 - a - b)
    for t in range(1, T):
        resid_prev = y[t - 1] - phi * (y[t - 2] if t >= 2 else 0.0)
        sigma2[t] = omega + a * resid_prev ** 2 + b * sigma2[t - 1]
        y[t] = phi * y[t - 1] + np.sqrt(sigma2[t]) * z[t]
    loc = np.concatenate([[0.0], phi * y[:-1]])
    return SyntheticSeries(y=y, loc=loc, scale=np.sqrt(sigma2), mixing_rate=abs(phi))


def ar1_heavytail(T, seed, phi=0.5, sigma=1.0, df=3.0):
    """AR(1) with standardised Student-t innovations: finite variance, heavy tails."""
    rng = _rng(seed)
    raw = stats.t.rvs(df=df, size=T, random_state=rng)
    eps = sigma * raw / np.sqrt(df / (df - 2))
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + eps[t]
    loc = np.concatenate([[0.0], phi * y[:-1]])
    return SyntheticSeries(y=y, loc=loc, scale=np.full(T, sigma), dist='t', df=df,
                           mixing_rate=abs(phi))


def changepoint_mean(T, seed, phi=0.5, sigma=1.0, n_changes=3, jump=4.0):
    """Abrupt level shifts. Static split CP recovers; the question is how fast."""
    rng = _rng(seed)
    cps = np.sort(rng.choice(np.arange(T // (n_changes + 2), T), size=n_changes, replace=False))
    level = np.zeros(T)
    signs = rng.choice([-1.0, 1.0], size=n_changes)
    for cp, s in zip(cps, signs):
        level[cp:] += s * jump
    eps = rng.normal(0, sigma, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = level[t] + phi * (y[t - 1] - level[t - 1]) + eps[t]
    loc = np.concatenate([[0.0], level[1:] + phi * (y[:-1] - level[:-1])])
    return SyntheticSeries(y=y, loc=loc, scale=np.full(T, sigma), changepoints=cps,
                           mixing_rate=abs(phi))


def changepoint_var(T, seed, phi=0.5, n_changes=3, lo=0.5, hi=3.0):
    """Abrupt variance shifts: the canonical motivation for adaptive thresholds."""
    rng = _rng(seed)
    cps = np.sort(rng.choice(np.arange(T // (n_changes + 2), T), size=n_changes, replace=False))
    scale = np.full(T, lo)
    bounds = np.concatenate([[0], cps, [T]])
    for i in range(len(bounds) - 1):
        scale[bounds[i]:bounds[i + 1]] = lo if i % 2 == 0 else hi
    eps = rng.normal(0, 1, T) * scale
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + eps[t]
    loc = np.concatenate([[0.0], phi * y[:-1]])
    return SyntheticSeries(y=y, loc=loc, scale=scale, changepoints=cps, mixing_rate=abs(phi))


def regime_switch(T, seed, p_stay=0.995, locs=(-2.0, 2.0), scales=(0.5, 2.0)):
    """Two-state Markov switching in both mean and variance."""
    rng = _rng(seed)
    state = np.zeros(T, dtype=int)
    for t in range(1, T):
        state[t] = state[t - 1] if rng.random() < p_stay else 1 - state[t - 1]
    loc = np.array(locs)[state]
    scale = np.array(scales)[state]
    y = loc + scale * rng.normal(0, 1, T)
    switches = np.flatnonzero(np.diff(state)) + 1
    # The regime is not observable from y alone, so the *observable* conditional law is a mixture;
    # `loc`/`scale` here condition on the regime and therefore give the strongest possible oracle.
    return SyntheticSeries(y=y, loc=loc, scale=scale, changepoints=switches,
                           exog=state[:, None].astype(float), exog_names=['regime'])


def slow_drift(T, seed, sigma=1.0):
    """Smoothly time-varying AR coefficient and scale: no changepoint to detect, just drift."""
    rng = _rng(seed)
    t_grid = np.linspace(0, 1, T)
    phi_t = 0.8 * np.sin(2 * np.pi * t_grid)
    scale = sigma * (0.5 + 1.5 * t_grid)
    y = np.zeros(T)
    eps = rng.normal(0, 1, T)
    for t in range(1, T):
        y[t] = phi_t[t] * y[t - 1] + scale[t] * eps[t]
    loc = np.concatenate([[0.0], phi_t[1:] * y[:-1]])
    return SyntheticSeries(y=y, loc=loc, scale=scale)


def seasonal_hetero(T, seed, period=24, sigma=1.0):
    """Seasonality and trend with season-dependent noise scale."""
    rng = _rng(seed)
    t_idx = np.arange(T)
    season = 3.0 * np.sin(2 * np.pi * t_idx / period)
    trend = 2.0 * t_idx / T
    scale = sigma * (0.5 + 1.0 * (1 + np.cos(2 * np.pi * t_idx / period)))
    loc = season + trend
    y = loc + scale * rng.normal(0, 1, T)
    exog = np.stack([np.sin(2 * np.pi * t_idx / period), np.cos(2 * np.pi * t_idx / period)], 1)
    return SyntheticSeries(y=y, loc=loc, scale=scale, exog=exog, exog_names=['sin', 'cos'])


def adversarial(T, seed, block=250):
    """Alternating blocks of wildly different scale and mean, with no smooth structure.

    Deliberately the worst case for exchangeability-based calibration; the regret-style metrics
    are the meaningful ones here.
    """
    rng = _rng(seed)
    n_blocks = int(np.ceil(T / block))
    block_loc = rng.normal(0, 5, n_blocks).repeat(block)[:T]
    block_scale = np.exp(rng.normal(0, 1, n_blocks)).repeat(block)[:T]
    y = block_loc + block_scale * rng.normal(0, 1, T)
    cps = np.arange(block, T, block)
    return SyntheticSeries(y=y, loc=block_loc, scale=block_scale, changepoints=cps)


def zaffran_ar09(T, seed, phi=0.9, sigma=1.0):
    """AgACI's AR(1) setting. Their `--ar -0.9` flag corresponds to phi = +0.9."""
    return ar1_gauss(T, seed, phi=phi, sigma=sigma)


def zaffran_arma(T, seed, phi=0.9, theta=0.5, sigma=1.0):
    """AgACI's ARMA(1,1) setting.

    The conditional law given the *observed* past is exact here because the innovations are
    recoverable recursively from y and the known parameters.
    """
    rng = _rng(seed)
    eps = rng.normal(0, sigma, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + eps[t] + theta * eps[t - 1]
    loc = np.concatenate([[0.0], phi * y[:-1] + theta * eps[:-1]])
    return SyntheticSeries(y=y, loc=loc, scale=np.full(T, sigma), mixing_rate=abs(phi))


GENERATORS = {
    'ar1_gauss': ar1_gauss,
    'ar1_garch': ar1_garch,
    'ar1_heavytail': ar1_heavytail,
    'changepoint_mean': changepoint_mean,
    'changepoint_var': changepoint_var,
    'regime_switch': regime_switch,
    'slow_drift': slow_drift,
    'seasonal_hetero': seasonal_hetero,
    'adversarial': adversarial,
    'zaffran_ar09': zaffran_ar09,
    'zaffran_arma': zaffran_arma,
}


def generate(name, T=6000, seed=0):
    if name not in GENERATORS:
        raise ValueError(f'Unknown synthetic generator {name}. Known: {sorted(GENERATORS)}')
    return GENERATORS[name](T, seed)
