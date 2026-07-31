"""
Selects which models, methods and hyperparameters to run.
Models, methods or hyperparameters are declared using HP(name=value) or HP(name=[value1, value2, ...]).
Join combines sets of hyperparameters in a cartesian product (similar to a grid search).
Union concatenates sets of hyperparameters.
"""

import collections

from moc.utils.hparams import HP, Join, Union


def get_default_tuning(config):
    default_methods = [
        'M-CP',
        'DR-CP',
        'C-HDR', 
        'HDR-H', 
        'PCP',
        'HD-PCP',
        'C-PCP',
        'CopulaCPTS',
    ]
    mqf2_methods = default_methods + [
        'STDQR',
        'L-CP',
        'L-H',
    ]

    # Posthoc grid
    posthoc_grid = Join(
        HP(method=default_methods),
    )
    posthoc_grid_mqf2 = Join(
        HP(method=mqf2_methods),
    )

    # DRF + KDE
    drf_kde = Join(
        HP(model='DRF-KDE'),
        HP(posthoc_grid=posthoc_grid)
    )

    # MQF2
    mqf2 = Join(
        HP(model='MQF2'),
        HP(posthoc_grid=posthoc_grid_mqf2)
    )

    # Mixture
    mixture = Join(
        HP(model='Mixture'),
        HP(mixture_size=[1, 10]),
        HP(posthoc_grid=posthoc_grid)
    )

    return Union(
        mixture,
        drf_kde,
        mqf2,
    )


def get_tuning_glow(config):
    methods = [
        'M-CP',
        'DR-CP',
        'C-HDR', 
        'HDR-H', 
        'PCP',
        'HD-PCP',
        'C-PCP',
        'STDQR',
        'L-CP',
        'L-H',
        'CopulaCPTS',
    ]

    posthoc_grid = Join(
        HP(method=methods),
    )

    glow = Join(
        HP(model='Glow'),
        HP(posthoc_grid=posthoc_grid)
    )

    return glow


def get_hparams_tuning(config):
    posthoc_grid_mqf2 = Union(
        Join(HP(method='M-CP'), HP(correction_factor=[0, 0.2, 0.4, 0.6, 0.8, 1])),
        Join(HP(method='C-HDR'), HP(n_samples=[5, 10, 30, 100, 300])),
        Join(HP(method='PCP'), HP(n_samples=[5, 10, 30, 100, 300])),
        Join(HP(method='HD-PCP'), HP(n_samples=[5, 10, 30, 100, 300])),
        Join(
            HP(method='C-PCP'), 
            Union(
                Join(HP(n_samples_mc=[5, 10, 30, 100, 300]), HP(n_samples_ref=[100])), 
                Join(HP(n_samples_mc=[100]), HP(n_samples_ref=[5, 10, 30, 300]))
            )
        ),
    )

    mqf2 = Join(
        HP(model='MQF2'),
        HP(posthoc_grid=posthoc_grid_mqf2)
    )

    return Union(
        mqf2,
    )


def get_larger_mqf2_tuning(config):
    default_methods = [
        'M-CP',
        'DR-CP',
        'C-HDR', 
        'HDR-H', 
        'PCP',
        'HD-PCP',
        'C-PCP',
    ]
    mqf2_methods = default_methods + [
        'L-CP',
        'L-H',
    ]

    # Posthoc grid
    posthoc_grid_mqf2 = Join(
        HP(method=mqf2_methods),
    )

    # MQF2
    mqf2 = Join(
        HP(model='MQF2'),
        HP(icnn_hidden_size=40),
        HP(icnn_num_layers=5),
        HP(estimate_logdet=[True, False]),
        HP(posthoc_grid=posthoc_grid_mqf2),
    )

    return Union(
        mqf2,
    )


POINT_SCORES = ['abs_residual', 'signed_residual', 'normalized_residual']
QUANTILE_SCORES = POINT_SCORES + ['cqr', 'cqr_r']
DENSITY_SCORES = POINT_SCORES + ['nll', 'hpd', 'pit']
ALL_TS_SCORES = sorted(set(QUANTILE_SCORES + DENSITY_SCORES))


def _ts_posthoc_grid(scores, full=False):
    """The score x scheme cross-product.

    Incompatible pairs are skipped at run time by `moc/models/ts_train.py`, so the same grid can be
    handed to every base model regardless of its `output_type`.
    """
    windows = [100, 500, 2000] if full else [500]
    rhos = [0.99, 0.995, 0.999] if full else [0.99]
    gammas = [0.001, 0.005, 0.01, 0.05] if full else [0.01]
    lrs = [0.05, 0.1, 0.5] if full else [0.1]

    return Union(
        Join(HP(method='Split'), HP(score=scores)),
        Join(HP(method='Rolling'), HP(score=scores), HP(window=windows)),
        Join(HP(method='NexCP'), HP(score=scores), HP(rho=rhos)),
        Join(HP(method='ACI'), HP(score=scores), HP(gamma=gammas)),
        Join(HP(method='DtACI'), HP(score=scores)),
        Join(HP(method='SF-OGD'), HP(score=scores)),
        Join(HP(method='SAOCP'), HP(score=scores)),
        Join(HP(method='QuantileTracker'), HP(score=scores), HP(lr=lrs)),
        Join(HP(method='PID'), HP(score=scores), HP(lr=lrs)),
    )


def get_timeseries_tuning(config, full=False):
    """Base model x score x calibration scheme.

    The score axis is what the existing time-series benchmarks hold fixed; making it a declared
    factor is the point of building this on top of the multi-output framework.
    """
    grid = _ts_posthoc_grid(ALL_TS_SCORES, full=full)

    point_models = Join(
        HP(model=['SeasonalNaive', 'Ridge', 'LGBM']),
        HP(posthoc_grid=grid),
    )
    quantile_models = Join(
        HP(model=['LGBMQuantile']),
        HP(posthoc_grid=grid),
    )
    density_models = Join(
        HP(model=['GaussianRidge', 'RFKDE']),
        HP(posthoc_grid=grid),
    )
    # Pairing the bootstrap ensemble with `Rolling` reproduces EnbPI; pairing it with the other
    # schemes gives combinations the literature has not reported.
    ensemble_models = Join(
        HP(model=['EnbPIEnsemble']),
        HP(posthoc_grid=grid),
    )
    return Union(point_models, quantile_models, density_models, ensemble_models)


def get_timeseries_test_tuning(config):
    """A single cheap model and a handful of schemes, for smoke tests."""
    grid = Union(
        Join(HP(method='Split'), HP(score=['abs_residual'])),
        Join(HP(method='Rolling'), HP(score=['abs_residual']), HP(window=[200])),
        Join(HP(method='ACI'), HP(score=['abs_residual']), HP(gamma=[0.01])),
    )
    return Join(HP(model=['Ridge']), HP(posthoc_grid=grid))


def get_timeseries_oracle_tuning(config):
    """Synthetic datasets only: the exact conditional law as the base model."""
    grid = _ts_posthoc_grid(DENSITY_SCORES, full=False)
    return Join(HP(model=['Oracle']), HP(posthoc_grid=grid))


def _get_tuning(config):
    if config.tuning_type == 'timeseries':
        return get_timeseries_tuning(config)
    elif config.tuning_type == 'timeseries_full':
        return get_timeseries_tuning(config, full=True)
    elif config.tuning_type == 'timeseries_test':
        return get_timeseries_test_tuning(config)
    elif config.tuning_type == 'timeseries_oracle':
        return get_timeseries_oracle_tuning(config)
    elif config.tuning_type == 'default':
        return get_default_tuning(config)
    elif config.tuning_type == 'larger_mqf2':
        return get_larger_mqf2_tuning(config)
    elif config.tuning_type == 'glow':
        return get_tuning_glow(config)
    elif config.tuning_type == 'hparams':
        return get_hparams_tuning(config)
    raise ValueError('Invalid tuning type')


def duplicates(choices):
    frozendict = lambda d: frozenset(d.items())
    frozen_choices = map(frozendict, choices)
    return [choice for choice, count in collections.Counter(frozen_choices).items() if count > 1]


def remove_duplicates(seq_of_dicts):
    seen = set()
    deduped_seq = []
    
    for d in seq_of_dicts:
        t = tuple(frozenset(d.items()))
        if t not in seen:
            seen.add(t)
            deduped_seq.append(d)
            
    return deduped_seq


def get_tuning(config):
    tuning = _get_tuning(config)
    tuning = remove_duplicates(tuning)
    dup = duplicates(tuning)
    assert len(dup) == 0, dup
    return tuning

