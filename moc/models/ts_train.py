"""
Per-run driver for the time-series testbed: the analogue of `moc/models/train.py::run`.

One run = one (dataset, base model, rolling-origin fold). Within it, every (score, scheme) pair in
the posthoc grid is evaluated against the same fitted model, which is what the score/threshold
decoupling in `TSEvaluator` buys.
"""

import logging
import traceback
from copy import copy

from moc.conformal.online import schemes
from moc.conformal.scores import compatible
from moc.datamodules.timeseries_datamodule import TimeSeriesDataModule
from moc.metrics.ts_evaluator import TSEvaluator
from moc.models.ts import ts_models

log = logging.getLogger('moc')


def build_model(model_name, hparams, datamodule):
    kwargs = dict(hparams)
    cls = ts_models[model_name]
    if model_name == 'Oracle':
        kwargs['datamodule'] = datamodule
    if model_name == 'SeasonalNaive':
        # The datamodule appends the seasonal lag right after the plain lags, when it made one.
        names = datamodule.feature_names
        seasonal = [i for i, n in enumerate(names) if n.startswith('lag_seasonal_')]
        kwargs['seasonal_index'] = seasonal[0] if seasonal else 0
    return cls(**kwargs)


def make_posthoc_rc(rc, model_hparams, posthoc_hparams, metrics):
    out = copy(rc)
    prefixed = {f'posthoc_{k}': v for k, v in posthoc_hparams.items()}
    out.hparams = {**model_hparams, **prefixed}
    out.metrics = dict(metrics)
    out.config = None      # the runner re-attaches this; keeps the pickles small
    return out


def run(rc, process_index=0):
    log.info(f'Starting {rc.summary_str()}')
    datamodule = TimeSeriesDataModule(rc)

    hparams = dict(rc.hparams)
    posthoc_grid = hparams.pop('posthoc_grid')
    model_name = hparams.pop('model')
    model_hparams = {'model': model_name, **hparams}

    model = build_model(model_name, hparams, datamodule)
    output_type = model.output_type()
    evaluator = TSEvaluator(
        datamodule, model, rc.config.alpha,
        grid_size=rc.config.ts.grid_size,
        local_windows=tuple(rc.config.ts.local_coverage_windows),
        wsc_directions=rc.config.ts.wsc_directions,
    )

    rcs = []
    for entry in posthoc_grid:
        entry = dict(entry)
        method = entry.pop('method')
        score = entry.pop('score')

        if not compatible(score, output_type):
            log.debug(f'skipping {score}/{method}: needs a richer model than {output_type}')
            continue
        scheme_cls = schemes[method]
        if getattr(scheme_cls, 'requires_model_refit', False) or getattr(scheme_cls, 'expensive', False):
            if not rc.config.ts.allow_sequential_model_methods:
                log.debug(f'skipping {method}: sequential model-in-the-loop methods are disabled')
                continue

        try:
            metrics = evaluator.evaluate(score, method, scheme_kwargs=entry)
        except Exception as e:
            log.warning(f'{rc.dataset}/{model_name}/{score}/{method} failed: {e}')
            log.debug(traceback.format_exc())
            continue
        rcs.append(make_posthoc_rc(
            rc, model_hparams, {'method': method, 'score': score, **entry}, metrics
        ))

    log.info(f'Finished {rc.summary_str()} ({len(rcs)} combinations)')
    return rc, rcs
