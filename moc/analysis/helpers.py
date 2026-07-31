main_metrics = ['coverage', 'median_region_size', 'region_size', 'cond_cov_x_error', 'cond_cov_z_error', 'wsc', 'test_coverage_time', 'score_time', 'total_time']
other_metrics = ['log_region_size', 'log_exact_region_size', 'q']

conformal_methods = ['M-CP', 'CopulaCPTS', 'DR-CP', 'C-HDR', 'PCP', 'HD-PCP', 'STDQR', 'C-PCP', 'L-CP']


def create_name_from_dict(d, config):
    name = d['posthoc_method']
    if config.name == 'hparams':
        if name == 'M-CP':
            correction_factor = d['posthoc_correction_factor']
            name = f'{name}-{correction_factor}'
        elif name in ['C-HDR', 'PCP', 'HD-PCP']:
            n_samples = d['posthoc_n_samples']
            name = f'{name}-{n_samples}'
        elif name == 'C-PCP':
            n_samples_mc = d['posthoc_n_samples_mc']
            n_samples_ref = d['posthoc_n_samples_ref']
            name = f'{name}-{n_samples_mc}'
            name = f'{name}-{n_samples_ref}'
    return name


# Time-series testbed metrics, grouped as validity / efficiency / compute.
ts_metric_names = {
    'coverage': 'Marginal coverage',
    'coverage_gap': 'Coverage gap',
    'lce_50': 'LCE (W=50)',
    'lce_100': 'LCE (W=100)',
    'lce_250': 'LCE (W=250)',
    'worst_window_100': 'Worst window (W=100)',
    'longest_miss_run': 'Longest miss run',
    'runs_test_p': 'Runs test $p$',
    'miscoverage_acf1': 'Miscoverage ACF(1)',
    'vol_cov_gap': 'CEC-vol',
    'vol_cov_worst': 'Worst volatility decile',
    'cp_coverage_after': 'Coverage after changepoint',
    'cp_recovery_steps': 'Recovery steps',
    'oracle_cond_cov': 'Oracle conditional coverage',
    'oracle_cond_cov_error': 'Oracle CEC',
    'width': 'Mean width',
    'median_width': 'Median width',
    'log_width': 'G. width',
    'width_normalized': 'Width / $\\sigma_y$',
    'winkler': 'Winkler score',
    'pinball_lo': 'Pinball ($\\alpha/2$)',
    'pinball_hi': 'Pinball ($1-\\alpha/2$)',
    'n_components': 'Components',
    'nonconvex_rate': 'Non-convex rate',
    'empty_rate': 'Empty rate',
    'infinite_rate': 'Infinite rate',
    'truncated_rate': 'Grid-truncated rate',
    'adaptivity_resid': 'Adaptivity (residual)',
    'adaptivity_oracle': 'Adaptivity (oracle)',
    'oracle_width_ratio': 'Width / oracle width',
    'regret': 'Regret',
    'update_time': 'Update time (s)',
    'fit_time': 'Fit time (s)',
    'score_acf1': 'Score ACF(1)',
    'score_acf_decay': 'Score ACF decay',
    'score_ljung_box_p': 'Ljung-Box $p$',
    'true_mixing_rate': 'True mixing rate',
}


def get_metric_name(metric):
    if metric in ts_metric_names:
        return ts_metric_names[metric]
    return {
        'coverage': 'Marginal coverage',
        'region_size': 'Mean Region Size',
        'log_region_size': 'G. Size',
        'cond_cov_x_error': 'CEC-$X$ (x100)',
        'cond_cov_z_error': 'CEC-$Z$ (x100)',
        'wsc': 'WSC',
        'score_time': 'Score Time',
        'test_coverage_time': 'Test coverage Time',
        'total_time': 'Time (s)',
        'log_exact_region_size': 'Geometric Mean Exact Region Size',
        'median_region_size': 'Median Region Size'
    }[metric]
