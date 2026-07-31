"""
Evaluation driver for the time-series testbed.

Implements the two-pass decoupling from docs/TIMESERIES_TESTBED_PLAN.md §3.2:

1. **Score pass** -- with the base model fixed, compute the conformity scores for the calibration
   and test blocks once, vectorised, and cache them. This is the model-bound, expensive part.
2. **Threshold pass** -- every online scheme consumes that cached stream and emits `q_t` in an
   O(T) numpy recursion with no model calls at all.

The consequence is that adding a scheme to the sweep costs milliseconds rather than a refit, which
is what makes the score x scheme cross-product affordable. `SPCI` opts out via
`requires_model_refit` and trains inside the loop.
"""

import logging
import time

import numpy as np

from moc.conformal.online import SCHEME_FAMILY, schemes
from moc.conformal.scores import compatible, scores as score_registry
from moc.metrics import ts_metrics as tsm
from moc.metrics.conditional_coverage_metrics import wsc_unbiased

log = logging.getLogger('moc')


class TSEvaluator:
    """Fits the base model once, then evaluates any number of (score, scheme) pairs against it."""

    def __init__(self, datamodule, model, alpha, grid_size=2000, grid_pad=0.5,
                 local_windows=(50, 100, 250), wsc_directions=100):
        self.dm = datamodule
        self.model = model
        self.alpha = alpha
        self.grid_size = grid_size
        self.grid_pad = grid_pad
        self.local_windows = tuple(local_windows)
        self.wsc_directions = wsc_directions

        self._score_cache = {}
        self._grid_cache = {}
        self.fit_time = 0.0
        self._fit_and_predict()

    # -- pass 1: the model-bound work, done once ---------------------------------------------

    def _fit_and_predict(self):
        dm = self.dm
        t0 = time.perf_counter()

        if getattr(self.model, 'provides_loo', False):
            # EnbPI's premise: no held-out calibration block. Fit on everything before the test
            # block and calibrate on out-of-bag residuals.
            x_fit = np.concatenate([dm.train.x, dm.val.x, dm.calib.x])
            y_fit = np.concatenate([dm.train.y, dm.val.y, dm.calib.y])
            self.model.fit(x_fit, y_fit)
            self.pred_calib = self.model.loo_predictive()
            self.y_calib = y_fit
            self.calib_split = None
        else:
            self.model.fit(dm.train.x, dm.train.y)
            self.pred_calib = self._predict(dm.calib)
            self.y_calib = dm.calib.y
            self.calib_split = dm.calib

        self.pred_test = self._predict(dm.test)
        self.y_test = dm.test.y
        self.fit_time = time.perf_counter() - t0

        # Scale for the normalised-residual score, taken from training residuals so that the
        # calibration block never informs its own normalisation.
        train_pred = self._predict(dm.train)
        self.init_scale = float(np.std(dm.train.y - train_pred.mean)) or 1.0
        self.test_residual = np.abs(self.y_test - self.pred_test.mean)

        lo = min(self.y_calib.min(), self.y_test.min())
        hi = max(self.y_calib.max(), self.y_test.max())
        pad = self.grid_pad * (hi - lo if hi > lo else 1.0)
        self.grid = np.linspace(lo - pad, hi + pad, self.grid_size)

    def _predict(self, split):
        if getattr(self.model, 'needs_split', False):
            return self.model.predict(split.x, split=split)
        return self.model.predict(split.x)

    def _scores(self, score_name, **score_kwargs):
        key = (score_name, tuple(sorted(score_kwargs.items())))
        if key not in self._score_cache:
            t0 = time.perf_counter()
            score = score_registry[score_name](**score_kwargs)
            s_calib, s_test = score.prepare(
                self.pred_calib, self.y_calib, self.pred_test, self.y_test,
                self.alpha, init_scale=self.init_scale, grid=self.grid,
            )
            grid_scores = score.grid_scores(self.grid)
            self._score_cache[key] = (score, s_calib, s_test, time.perf_counter() - t0)
            self._grid_cache[key] = grid_scores
        score, s_calib, s_test, score_time = self._score_cache[key]
        return score, s_calib, s_test, self._grid_cache[key], score_time

    # -- pass 2: the cheap part ----------------------------------------------------------------

    def evaluate(self, score_name, scheme_name, score_kwargs=None, scheme_kwargs=None):
        score_kwargs = score_kwargs or {}
        scheme_kwargs = scheme_kwargs or {}
        output_type = self.model.output_type()
        if not compatible(score_name, output_type):
            raise ValueError(
                f'score {score_name!r} needs a richer model than output_type={output_type!r}'
            )

        score, s_calib, s_test, grid_scores, score_time = self._scores(score_name, **score_kwargs)
        alphas = score.stream_alphas(self.alpha)

        scheme = schemes[scheme_name](**scheme_kwargs)
        t0 = time.perf_counter()
        q = scheme.thresholds(s_calib, s_test, alphas)
        update_time = time.perf_counter() - t0

        covered = np.all(s_test <= q, axis=1)
        membership = np.all(grid_scores <= q[:, None, :], axis=2)

        metrics = self._metrics(covered, membership, q, s_test, scheme)
        metrics['score_time'] = score_time
        metrics['update_time'] = update_time
        metrics['fit_time'] = self.fit_time
        metrics['total_time'] = score_time + update_time + self.fit_time
        metrics['scheme_family'] = SCHEME_FAMILY.get(scheme_name, 'unknown')
        return metrics

    def _metrics(self, covered, membership, q, s_test, scheme):
        target = 1 - self.alpha
        geom = tsm.set_geometry(membership, self.grid)
        width, lower, upper = geom['width'], geom['lower'], geom['upper']

        m = {'coverage': float(covered.mean()), 'coverage_gap': float(covered.mean() - target)}
        m.update(tsm.local_coverage_metrics(covered, target, self.local_windows))
        m.update(tsm.miscoverage_runs(covered))

        finite = np.isfinite(width) & ~geom['empty']
        m['width'] = float(np.mean(width))
        m['median_width'] = float(np.median(width))
        m['log_width'] = float(np.mean(np.log(np.maximum(width, 1e-12))))
        m['width_normalized'] = float(np.mean(width) / self.dm.y_scale)
        m['n_components'] = float(np.mean(geom['n_components']))
        m['nonconvex_rate'] = float(np.mean(geom['n_components'] > 1))
        m['empty_rate'] = float(np.mean(geom['empty']))
        m['truncated_rate'] = float(np.mean(geom['truncated']))
        m['infinite_rate'] = float(np.mean(~np.isfinite(q).all(axis=1)))

        if finite.any():
            m['winkler'] = float(np.mean(
                tsm.winkler_score(self.y_test[finite], lower[finite], upper[finite], self.alpha)
            ))
            m['pinball_lo'] = float(np.mean(
                tsm.pinball_at(self.y_test[finite], lower[finite], self.alpha / 2)))
            m['pinball_hi'] = float(np.mean(
                tsm.pinball_at(self.y_test[finite], upper[finite], 1 - self.alpha / 2)))
        else:
            m['winkler'] = m['pinball_lo'] = m['pinball_hi'] = np.nan

        m['adaptivity_resid'] = tsm.adaptivity(width, self.test_residual)
        m['regret'] = tsm.regret_vs_best_fixed(s_test[:, 0], q[:, 0], self.alpha)

        # Conditional coverage: worst-slab over the lag/exogenous features, reusing the existing
        # implementation from the multi-output pipeline unchanged.
        try:
            m['wsc'] = float(wsc_unbiased(
                self.dm.test.x, covered.astype(float), delta=0.2, M=self.wsc_directions
            ))
        except Exception as e:
            log.debug(f'wsc failed: {e}')
            m['wsc'] = np.nan

        vol = _rolling_volatility(self.test_residual)
        m.update(tsm.stratified_coverage(covered, vol, prefix='vol'))
        m.update(tsm.changepoint_response(covered, self.dm.changepoints, target))

        if self.dm.test.oracle_loc is not None:
            m['adaptivity_oracle'] = tsm.adaptivity(width, self.dm.test.oracle_scale)
            m.update(tsm.oracle_conditional_coverage(
                lower, upper, self.dm.test.oracle_loc, self.dm.test.oracle_scale,
                self.dm.oracle_dist, self.dm.oracle_df, target,
            ))
        else:
            m['adaptivity_oracle'] = np.nan
            m.update({'oracle_cond_cov': np.nan, 'oracle_cond_cov_error': np.nan,
                      'oracle_width_ratio': np.nan})

        m.update(tsm.dependence_diagnostics(s_test[:, 0]))
        if self.dm.mixing_rate is not None:
            m['true_mixing_rate'] = float(self.dm.mixing_rate)
        else:
            m['true_mixing_rate'] = np.nan

        m['n_test'] = int(len(covered))
        m['scheme_desc'] = scheme.describe()
        return m


def _rolling_volatility(residual, window=25):
    """Causal rolling scale of the residual, used to stratify coverage by local difficulty."""
    r = np.abs(np.asarray(residual, dtype=float))
    out = np.full(len(r), np.nan)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    for t in range(len(r)):
        lo = max(0, t - window)
        if t - lo >= 5:
            out[t] = (csum[t] - csum[lo]) / (t - lo)
    return out
