"""
Analysis helpers for the time-series testbed.

Deliberately thin: the point of building on this repository is that the pickled-RunConfig ->
tidy-dataframe -> critical-difference-diagram stack already exists. This module only reshapes the
time-series runs into the long format `moc/analysis/plot_cd_diagram.py::draw_my_cd_diagram`
expects, and groups metrics under the validity / efficiency / compute headings used by
Stocker et al. (2025).
"""

import numpy as np
import pandas as pd

from moc.analysis.dataframes import load_df

# Metrics grouped as in the prior review, so results tables compose with it.
VALIDITY_METRICS = [
    'coverage', 'coverage_gap', 'lce_50', 'lce_100', 'lce_250',
    'worst_window_100', 'longest_miss_run', 'runs_test_p', 'miscoverage_acf1',
    'wsc', 'vol_cov_gap', 'vol_cov_worst',
    'cp_coverage_after', 'cp_recovery_steps',
    'oracle_cond_cov', 'oracle_cond_cov_error',
]
EFFICIENCY_METRICS = [
    'width', 'median_width', 'log_width', 'width_normalized', 'winkler',
    'pinball_lo', 'pinball_hi', 'n_components', 'nonconvex_rate',
    'empty_rate', 'infinite_rate', 'truncated_rate',
    'adaptivity_resid', 'adaptivity_oracle', 'oracle_width_ratio', 'regret',
]
COMPUTE_METRICS = ['score_time', 'update_time', 'fit_time', 'total_time']
DIAGNOSTIC_METRICS = ['score_acf1', 'score_acf_decay', 'score_ljung_box_p', 'true_mixing_rate']

TS_METRICS = VALIDITY_METRICS + EFFICIENCY_METRICS + COMPUTE_METRICS + DIAGNOSTIC_METRICS

METRIC_GROUP = (
    {m: 'validity' for m in VALIDITY_METRICS}
    | {m: 'efficiency' for m in EFFICIENCY_METRICS}
    | {m: 'compute' for m in COMPUTE_METRICS}
    | {m: 'diagnostic' for m in DIAGNOSTIC_METRICS}
)

# Metrics where being closer to the nominal level is better, rather than lower being better.
CENTRED_METRICS = {'coverage', 'wsc', 'oracle_cond_cov', 'cp_coverage_after'}

HPARAM_COLUMNS = ['model', 'posthoc_method', 'posthoc_score', 'posthoc_window', 'posthoc_rho',
                  'posthoc_gamma', 'posthoc_lr']


def load_ts_results(config, reload=True):
    """Wide dataframe: one row per (dataset, fold, model, score, scheme)."""
    df = load_df(config, reload=reload)
    for col in HPARAM_COLUMNS:
        df[col] = df.apply(lambda r: r.hparams.get(col, None), axis=1)
    for metric in TS_METRICS:
        df[metric] = df.apply(lambda r: r.metrics.get(metric, np.nan), axis=1)
    df['scheme_family'] = df.apply(lambda r: r.metrics.get('scheme_family', None), axis=1)
    df = df.rename(columns={'posthoc_method': 'scheme', 'posthoc_score': 'score'})
    df = df.drop(columns=['hparams', 'metrics', 'config'], errors='ignore')
    return df.reset_index(drop=True)


def method_name(df, by=('score', 'scheme')):
    """The label that becomes one tick on a critical-difference diagram."""
    return df[list(by)].astype(str).agg(' / '.join, axis=1)


def to_long(df, metrics=None, by=('score', 'scheme')):
    """Long format with the columns `draw_my_cd_diagram` expects: dataset, name, metric, value."""
    metrics = list(metrics or TS_METRICS)
    out = df.copy()
    out['name'] = method_name(out, by)
    present = [m for m in metrics if m in out.columns]
    long = out.melt(
        id_vars=['dataset_group', 'dataset', 'run_id', 'name', 'model', 'score', 'scheme'],
        value_vars=present, var_name='metric', value_name='value',
    )
    return long


def summary_table(df, metrics=('coverage', 'lce_100', 'width', 'winkler', 'total_time'),
                  by=('score', 'scheme')):
    """Mean and standard deviation over folds and datasets, one row per method."""
    out = df.copy()
    out['name'] = method_name(out, by)
    agg = out.groupby('name', dropna=False)[list(metrics)].agg(['mean', 'std'])
    return agg.sort_values((metrics[0], 'mean'))


def rank_table(df, metric, alpha=0.1, by=('score', 'scheme')):
    """Mean rank per dataset, the input a CD diagram summarises.

    Lower is better after the transformation, matching `draw_my_cd_diagram`: coverage-like metrics
    are folded to |value - (1 - alpha)| first, since overshooting the nominal level is not a win.
    """
    out = df.copy()
    out['name'] = method_name(out, by)
    per_dataset = out.groupby(['dataset', 'name'], dropna=False, observed=True)[metric].mean()
    per_dataset = per_dataset.reset_index()
    if metric in CENTRED_METRICS:
        per_dataset[metric] = (per_dataset[metric] - (1 - alpha)).abs()
    pivot = per_dataset.pivot(index='dataset', columns='name', values=metric)
    return pivot.rank(axis=1).mean().sort_values()


def validity_efficiency_frontier(df, validity='lce_100', efficiency='width',
                                 by=('score', 'scheme')):
    """The trade-off the review is really about, in one table.

    A method is on the frontier if nothing else is better on both axes at once.
    """
    out = df.copy()
    out['name'] = method_name(out, by)
    agg = out.groupby('name', dropna=False)[[validity, efficiency]].mean().dropna()
    dominated = np.zeros(len(agg), dtype=bool)
    values = agg.to_numpy()
    for i in range(len(agg)):
        strictly_better = (values <= values[i]).all(axis=1) & (values < values[i]).any(axis=1)
        dominated[i] = strictly_better.any()
    agg['on_frontier'] = ~dominated
    return agg.sort_values(validity)


def coverage_over_time(df_run_metrics):
    """Placeholder hook for the coverage-ribbon figures; the per-step series are not persisted by
    default because they dominate the pickle size. Re-run a single configuration with
    `TSEvaluator` directly to obtain them."""
    raise NotImplementedError(
        'Per-step trajectories are not stored in the run pickles. Instantiate TSEvaluator directly '
        'for a single configuration to plot coverage over time.'
    )
