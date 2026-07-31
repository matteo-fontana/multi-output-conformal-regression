"""
Metrics for univariate time-series conformal prediction.

Organised under the validity / efficiency / compute triad of Stocker, Małgorzewicz, Fontana &
Ben Taieb (2025) so that the results tables compose with that paper's.

Marginal coverage alone is nearly uninformative here -- every method in the study is designed to
hit it. What separates them is *when* the misses happen, whether they cluster, and what the sets
cost. The clustering diagnostics are also the empirical counterpart of the weak-dependence
conditions in the prior review's theory section.
"""

import numpy as np
from scipy import stats


# ------------------------------------------------------------------------------------------------
# Validity
# ------------------------------------------------------------------------------------------------

def rolling_coverage(covered, window):
    """Mean coverage in each sliding window of length `window`."""
    covered = np.asarray(covered, dtype=float)
    if len(covered) < window:
        return np.array([covered.mean()]) if len(covered) else np.array([np.nan])
    csum = np.concatenate([[0.0], np.cumsum(covered)])
    return (csum[window:] - csum[:-window]) / window


def local_coverage_metrics(covered, target, windows=(50, 100, 250)):
    """RMS and worst-case deviation of rolling coverage from the nominal level."""
    out = {}
    for w in windows:
        if len(covered) < w:
            out[f'lce_{w}'] = np.nan
            out[f'worst_window_{w}'] = np.nan
            continue
        rc = rolling_coverage(covered, w)
        dev = rc - target
        out[f'lce_{w}'] = float(np.sqrt(np.mean(dev ** 2)))
        out[f'worst_window_{w}'] = float(np.max(np.abs(dev)))
    return out


def miscoverage_runs(covered):
    """Clustering of the miscoverage sequence.

    Under exchangeability the miss indicators are (asymptotically) i.i.d. Bernoulli, so the runs
    test should not reject. Serial dependence in the scores shows up here long before it shows up
    in marginal coverage, which is why this is the diagnostic worth reporting.
    """
    miss = (~np.asarray(covered, dtype=bool)).astype(int)
    n = len(miss)
    out = {'longest_miss_run': 0.0, 'runs_test_z': np.nan, 'runs_test_p': np.nan,
           'miscoverage_acf1': np.nan}
    if n == 0:
        return out

    # Longest run of consecutive misses.
    longest, current = 0, 0
    for m in miss:
        current = current + 1 if m else 0
        longest = max(longest, current)
    out['longest_miss_run'] = float(longest)

    n1 = int(miss.sum())
    n0 = n - n1
    if n1 > 0 and n0 > 0 and n > 1:
        runs = 1 + int(np.sum(miss[1:] != miss[:-1]))
        mean = 2.0 * n1 * n0 / n + 1
        var = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n ** 2 * (n - 1))
        if var > 0:
            z = (runs - mean) / np.sqrt(var)
            out['runs_test_z'] = float(z)
            out['runs_test_p'] = float(2 * stats.norm.sf(abs(z)))
    if n > 2 and miss.std() > 0:
        out['miscoverage_acf1'] = float(np.corrcoef(miss[:-1], miss[1:])[0, 1])
    return out


def stratified_coverage(covered, strat_values, n_bins=10, prefix='vol'):
    """Coverage within bins of a stratifying variable (e.g. realised volatility deciles)."""
    covered = np.asarray(covered, dtype=float)
    v = np.asarray(strat_values, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() < n_bins * 2:
        return {f'{prefix}_cov_gap': np.nan, f'{prefix}_cov_worst': np.nan}
    edges = np.quantile(v[finite], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return {f'{prefix}_cov_gap': np.nan, f'{prefix}_cov_worst': np.nan}
    idx = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
    covs, weights = [], []
    for b in range(len(edges) - 1):
        mask = (idx == b) & finite
        if mask.sum() == 0:
            continue
        covs.append(covered[mask].mean())
        weights.append(mask.sum())
    covs, weights = np.array(covs), np.array(weights, dtype=float)
    weights /= weights.sum()
    return {
        f'{prefix}_cov_gap': float(np.sqrt(np.sum(weights * (covs - covered.mean()) ** 2))),
        f'{prefix}_cov_worst': float(np.max(np.abs(covs - covered.mean()))),
    }


def changepoint_response(covered, changepoints, target, window=100):
    """Coverage around known regime changes, and how long it takes to recover."""
    if changepoints is None or len(changepoints) == 0:
        return {'cp_coverage_after': np.nan, 'cp_recovery_steps': np.nan}
    covered = np.asarray(covered, dtype=float)
    n = len(covered)
    after, recovery = [], []
    for cp in np.atleast_1d(changepoints):
        cp = int(cp)
        if cp < 0 or cp >= n:
            continue
        seg = covered[cp:cp + window]
        if len(seg) == 0:
            continue
        after.append(seg.mean())
        # Steps until the trailing coverage re-enters a band around the nominal level.
        steps = np.nan
        for k in range(20, min(window, n - cp)):
            if abs(covered[cp:cp + k].mean() - target) < 0.05:
                steps = k
                break
        recovery.append(steps)
    if not after:
        return {'cp_coverage_after': np.nan, 'cp_recovery_steps': np.nan}
    return {
        'cp_coverage_after': float(np.mean(after)),
        'cp_recovery_steps': float(np.nanmean(recovery)) if np.any(np.isfinite(recovery)) else np.nan,
    }


def oracle_conditional_coverage(lower, upper, oracle_loc, oracle_scale, dist, df, target):
    """Exact conditional coverage of the realised interval under the known conditional law.

    Only available on the synthetic group, and the only measurement here that is a true
    conditional-coverage error rather than a proxy.
    """
    if oracle_loc is None:
        return {'oracle_cond_cov': np.nan, 'oracle_cond_cov_error': np.nan,
                'oracle_width_ratio': np.nan}
    if dist == 't':
        rv, std = stats.t(df=df), np.sqrt(df / (df - 2))
    else:
        rv, std = stats.norm, 1.0
    z_lo = (lower - oracle_loc) / oracle_scale * std
    z_hi = (upper - oracle_loc) / oracle_scale * std
    cond_cov = rv.cdf(z_hi) - rv.cdf(z_lo)
    alpha = 1 - target
    oracle_width = oracle_scale * 2 * rv.ppf(1 - alpha / 2) / std
    finite = np.isfinite(upper - lower)
    return {
        'oracle_cond_cov': float(np.nanmean(cond_cov)),
        'oracle_cond_cov_error': float(np.sqrt(np.nanmean((cond_cov - target) ** 2))),
        'oracle_width_ratio': float(np.nanmean((upper - lower)[finite] / oracle_width[finite]))
        if finite.any() else np.nan,
    }


def dependence_diagnostics(scores, max_lag=20):
    """Autocorrelation structure of the score stream.

    The bridge to the finite-sample split-conformal bounds under weak dependence: those bounds are
    stated in terms of how fast dependence decays, so the decay is worth measuring on every
    dataset rather than assumed.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    out = {'score_acf1': np.nan, 'score_acf_decay': np.nan, 'score_ljung_box_p': np.nan}
    n = len(s)
    if n < max_lag + 10 or s.std() == 0:
        return out
    s = s - s.mean()
    denom = np.dot(s, s)
    acf = np.array([np.dot(s[:-k], s[k:]) / denom for k in range(1, max_lag + 1)])
    out['score_acf1'] = float(acf[0])
    # Geometric decay rate fitted to |acf|, i.e. the rho in an assumed rho^k envelope.
    #
    # Only meaningful when there is autocorrelation to decay: for white noise the sample ACF is
    # already indistinguishable from zero at every lag, and fitting an exponential to that noise
    # returns a rate near 1, which would read as "very slow decay" -- the opposite of the truth.
    # Reporting NaN keeps an uninformative number out of the tables.
    band = 2.0 / np.sqrt(n)
    if np.abs(acf[0]) < band:
        out['score_acf_decay'] = np.nan
    else:
        significant = np.abs(acf) >= band
        k = int(np.argmin(significant)) if not significant.all() else max_lag
        k = max(k, 2)
        lags = np.arange(1, k + 1)
        slope = np.polyfit(lags, np.log(np.maximum(np.abs(acf[:k]), 1e-8)), 1)[0]
        out['score_acf_decay'] = float(np.exp(slope))
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(s, lags=[min(10, max_lag)], return_df=True)
        out['score_ljung_box_p'] = float(lb['lb_pvalue'].iloc[0])
    except Exception:
        pass
    return out


# ------------------------------------------------------------------------------------------------
# Efficiency
# ------------------------------------------------------------------------------------------------

def set_geometry(membership, grid):
    """Exact Lebesgue measure and connected-component count from grid membership.

    In one dimension there is no need for the Monte-Carlo volume estimator the multi-output code
    must use: integrating the indicator over a fine grid is exact to the grid resolution and,
    unlike an interval-only evaluation, correctly handles the unions of intervals produced by the
    density-based scores.
    """
    membership = np.asarray(membership, dtype=bool)
    dx = float(grid[1] - grid[0])
    n = membership.shape[0]

    width = membership.sum(axis=1) * dx
    # Components: count rising edges.
    padded = np.concatenate([np.zeros((n, 1), dtype=bool), membership], axis=1)
    n_components = (padded[:, 1:] & ~padded[:, :-1]).sum(axis=1)

    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    any_true = membership.any(axis=1)
    first = np.argmax(membership, axis=1)
    last = membership.shape[1] - 1 - np.argmax(membership[:, ::-1], axis=1)
    lower[any_true] = grid[first[any_true]]
    upper[any_true] = grid[last[any_true]]

    truncated = membership[:, 0] | membership[:, -1]
    return {
        'width': width, 'n_components': n_components.astype(float),
        'lower': lower, 'upper': upper, 'empty': ~any_true, 'truncated': truncated,
    }


def winkler_score(y, lower, upper, alpha):
    """Interval score: width plus a penalty proportional to how far outside the interval y falls."""
    y = np.asarray(y, dtype=float)
    width = upper - lower
    penalty = np.where(y < lower, 2 / alpha * (lower - y), 0.0) \
        + np.where(y > upper, 2 / alpha * (y - upper), 0.0)
    return width + penalty


def pinball_at(y, q, level):
    return np.maximum(level * (y - q), (1 - level) * (q - y))


def adaptivity(width, reference):
    """Rank correlation between set size and a measure of true local difficulty."""
    w = np.asarray(width, dtype=float)
    r = np.asarray(reference, dtype=float)
    mask = np.isfinite(w) & np.isfinite(r)
    if mask.sum() < 10 or np.std(w[mask]) == 0 or np.std(r[mask]) == 0:
        return np.nan
    return float(stats.spearmanr(w[mask], r[mask]).statistic)


def regret_vs_best_fixed(scores, q, alpha):
    """Excess pinball loss over the best constant threshold chosen in hindsight.

    The yardstick the online-learning half of the literature actually optimises.
    """
    s = np.asarray(scores, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    mask = np.isfinite(s) & np.isfinite(q)
    if mask.sum() < 10:
        return np.nan
    s, q = s[mask], q[mask]
    level = 1 - alpha
    ours = pinball_at(s, q, level).mean()
    best = pinball_at(s, np.quantile(s, level), level).mean()
    return float(ours - best)
