"""
Tests for the time-series conformal testbed.

Organised around the correctness traps listed in docs/TIMESERIES_TESTBED_PLAN.md §8, because those
are the failure modes that produce plausible-looking but wrong numbers rather than crashes.
"""

import numpy as np
import pytest

from moc.configs.config import get_config
from moc.conformal.online import schemes
from moc.conformal.online.base import conformal_quantile, weighted_conformal_quantile
from moc.conformal.scores import SCORES_BY_OUTPUT_TYPE, scores as score_registry
from moc.datamodules.timeseries_datamodule import TimeSeriesDataModule, make_windows
from moc.metrics import ts_metrics as tsm
from moc.metrics.ts_evaluator import TSEvaluator
from moc.models.ts import ts_models

ALPHA = 0.1


def make_rc(dataset='ar1_gauss', group='ts_synthetic', run_id=0, **overrides):
    config = get_config()
    config.alpha = ALPHA
    config.ts.synthetic_length = 3000
    for k, v in overrides.items():
        config.ts[k] = v
    from moc.utils.run_config import RunConfig
    return RunConfig(config=config, dataset_group=group, dataset=dataset, run_id=run_id)


@pytest.fixture(scope='module')
def dm():
    return TimeSeriesDataModule(make_rc('ar1_gauss'))


# ------------------------------------------------------------------------------------------------
# Guard 1: no future leakage
# ------------------------------------------------------------------------------------------------

def test_splits_are_chronological_and_gapped(dm):
    assert dm.train.t_index[-1] < dm.val.t_index[0]
    assert dm.val.t_index[-1] < dm.calib.t_index[0]
    assert dm.calib.t_index[-1] < dm.test.t_index[0]
    gap = dm.lags + dm.horizon - 1
    assert dm.calib.t_index[0] - dm.train.t_index[-1] > gap


def test_window_targets_never_appear_as_later_features(dm):
    """A window in the test block must not use any target from the calibration block."""
    earliest_test_feature = dm.test.t_index[0] - dm.lags
    assert earliest_test_feature > dm.calib.t_index[-1]


def test_windows_align_lags_with_targets():
    y = np.arange(100, dtype=float)
    x, target, t_index, names = make_windows(y, None, lags=3, horizon=1, seasonal_period=1)
    assert names[:3] == ['lag_1', 'lag_2', 'lag_3']
    np.testing.assert_allclose(x[:, 0], target - 1)
    np.testing.assert_allclose(x[:, 2], target - 3)


def test_scalers_use_training_block_only(dm):
    """Feature standardisation must be fit on train, so the train block is exactly centred."""
    np.testing.assert_allclose(dm.train.x.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(dm.train.x.std(axis=0), 1, atol=1e-6)
    # The later blocks are transformed with the same constants, so they are generally not centred.
    assert np.abs(dm.test.x.mean(axis=0)).max() > 1e-8


def test_horizon_above_one_is_refused():
    with pytest.raises(NotImplementedError, match='horizon=1'):
        make_windows(np.arange(100, dtype=float), None, lags=3, horizon=4, seasonal_period=1)


# ------------------------------------------------------------------------------------------------
# Guard 3: quantile primitives
# ------------------------------------------------------------------------------------------------

def test_conformal_quantile_is_the_right_order_statistic():
    s = np.arange(1, 20, dtype=float)          # n = 19, so ceil(0.9 * 20) = 18
    assert conformal_quantile(s, 0.1) == 18.0


def test_conformal_quantile_returns_inf_when_too_few_scores():
    assert conformal_quantile(np.arange(5, dtype=float), 0.1) == np.inf
    assert conformal_quantile(np.array([]), 0.1) == np.inf


def test_weighted_quantile_matches_unweighted_with_uniform_weights():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    w = np.ones(500)
    assert weighted_conformal_quantile(s, w, 0.1) == pytest.approx(conformal_quantile(s, 0.1))


def test_weighted_quantile_has_an_infinite_atom():
    """The `+1` in the normalisation is what keeps the procedure conservative; without it the
    weighted quantile would never return +inf."""
    s = np.arange(5, dtype=float)
    assert weighted_conformal_quantile(s, np.ones(5), 0.01) == np.inf


def test_decaying_weights_track_recent_scores():
    """A recent burst too small to move the unweighted quantile must still move the weighted one."""
    s = np.concatenate([np.zeros(590), np.full(10, 10.0)])
    age = np.arange(len(s) - 1, -1, -1)
    assert conformal_quantile(s, 0.1) == 0.0
    assert weighted_conformal_quantile(s, 0.9 ** age, 0.1) == 10.0


# ------------------------------------------------------------------------------------------------
# Guard 4: the online loop must be causal
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('scheme_name', sorted(schemes))
def test_thresholds_do_not_depend_on_future_scores(scheme_name):
    """q_t may use scores strictly before t and nothing else.

    Perturbing the tail of the score stream must leave every earlier threshold untouched. This is
    the test that catches an off-by-one in the update, which would otherwise show up only as
    suspiciously good coverage.
    """
    rng = np.random.default_rng(1)
    s_calib = np.abs(rng.normal(size=200))
    s_test = np.abs(rng.normal(size=120))
    k = 60

    kwargs = {'stride': 5, 'window': 100} if scheme_name == 'SPCI' else {}
    if scheme_name == 'PID+Scorecaster':
        kwargs = {'scorecast_stride': 40}

    q_ref = schemes[scheme_name](**kwargs).thresholds(s_calib, s_test, [0.1])
    s_perturbed = s_test.copy()
    s_perturbed[k:] += 100.0
    q_new = schemes[scheme_name](**kwargs).thresholds(s_calib, s_perturbed, [0.1])

    np.testing.assert_allclose(q_ref[:k + 1], q_new[:k + 1], atol=1e-10)


@pytest.mark.parametrize('scheme_name', sorted(schemes))
def test_every_scheme_produces_usable_thresholds(scheme_name):
    rng = np.random.default_rng(2)
    s_calib = np.abs(rng.normal(size=300))
    s_test = np.abs(rng.normal(size=200))
    kwargs = {'stride': 20} if scheme_name == 'SPCI' else {}
    if scheme_name == 'PID+Scorecaster':
        kwargs = {'scorecast_stride': 100}
    q = schemes[scheme_name](**kwargs).thresholds(s_calib, s_test, [0.1])
    assert q.shape == (200, 1)
    assert np.isfinite(q).mean() > 0.5


def test_two_stream_scores_get_one_threshold_per_stream():
    rng = np.random.default_rng(3)
    s_calib = rng.normal(size=(300, 2))
    s_test = rng.normal(size=(200, 2))
    q = schemes['Split']().thresholds(s_calib, s_test, [0.05, 0.05])
    assert q.shape == (200, 2)
    assert q[0, 0] != q[0, 1]


# ------------------------------------------------------------------------------------------------
# Guards 6 and 7: degenerate sets and grid adequacy
# ------------------------------------------------------------------------------------------------

def test_aci_counts_degenerate_steps():
    """When alpha_t leaves [0, 1] the prescription is an infinite or empty set. Those steps must be
    counted, not silently folded into the mean width."""
    scheme = schemes['ACI'](gamma=0.9)
    s_calib = np.abs(np.random.default_rng(4).normal(size=100))
    s_test = np.abs(np.random.default_rng(5).normal(size=300))
    q = scheme.thresholds(s_calib, s_test, [0.1])
    assert scheme.n_infinite + scheme.n_empty > 0
    assert np.isinf(q).sum() == scheme.n_infinite + scheme.n_empty


def test_evaluation_grid_contains_every_realised_observation(dm):
    ev = TSEvaluator(dm, ts_models['Ridge'](), ALPHA, grid_size=500)
    assert ev.grid[0] < ev.y_test.min()
    assert ev.grid[-1] > ev.y_test.max()


def test_grid_membership_agrees_with_score_based_coverage(dm):
    """The two coverage computations are independent paths through the code; they must agree up to
    the grid resolution."""
    ev = TSEvaluator(dm, ts_models['GaussianRidge'](), ALPHA, grid_size=4000)
    for score_name in ['abs_residual', 'signed_residual', 'nll', 'pit']:
        score, s_calib, s_test, grid_scores, _ = ev._scores(score_name)
        q = schemes['Split']().thresholds(s_calib, s_test, score.stream_alphas(ALPHA))
        cov_score = np.all(s_test <= q, axis=1)
        membership = np.all(grid_scores <= q[:, None, :], axis=2)
        nearest = np.abs(ev.grid[None, :] - ev.y_test[:, None]).argmin(axis=1)
        cov_grid = membership[np.arange(len(ev.y_test)), nearest]
        assert np.mean(cov_score != cov_grid) < 0.01, score_name


# ------------------------------------------------------------------------------------------------
# Statistical behaviour
# ------------------------------------------------------------------------------------------------

def test_split_conformal_is_valid_on_exchangeable_data():
    """With i.i.d. data and a correctly specified model, split conformal must hit the nominal
    level: the sanity check that the plumbing does not systematically bias coverage."""
    rng = np.random.default_rng(6)
    s_calib = np.abs(rng.normal(size=2000))
    covs = []
    for _ in range(40):
        s_test = np.abs(rng.normal(size=500))
        q = schemes['Split']().thresholds(s_calib, s_test, [0.1])
        covs.append(np.mean(s_test <= q[:, 0]))
    assert 0.88 < np.mean(covs) < 0.92


@pytest.mark.parametrize('scheme_name', ['ACI', 'SAOCP', 'PID'])
def test_adaptive_schemes_beat_split_under_drift(scheme_name):
    """The behavioural claim the whole family rests on.

    `slow_drift` grows the noise scale monotonically, so a threshold calibrated on the calibration
    block is systematically wrong on the test block and cannot be rescued by luck. Note this is a
    claim about *drift*, not about changepoints: on an abrupt variance shift, ACI's oscillation can
    leave its worst-window deviation larger than static split conformal's even as its marginal
    coverage improves, which is itself worth reporting rather than asserting away.
    """
    dm = TimeSeriesDataModule(make_rc('slow_drift'))
    ev = TSEvaluator(dm, ts_models['Ridge'](), ALPHA, grid_size=600)
    split = ev.evaluate('abs_residual', 'Split')
    adaptive = ev.evaluate('abs_residual', scheme_name)
    assert abs(adaptive['coverage_gap']) < abs(split['coverage_gap'])
    assert adaptive['lce_100'] < split['lce_100']


def test_oracle_model_attains_the_best_conditional_coverage():
    """On synthetic data the exact conditional law should dominate a misspecified one on the
    oracle conditional-coverage error -- the upper bound the synthetic group exists to provide."""
    dm = TimeSeriesDataModule(make_rc('ar1_garch'))
    oracle = TSEvaluator(dm, ts_models['Oracle'](datamodule=dm), ALPHA, grid_size=800)
    plain = TSEvaluator(dm, ts_models['Ridge'](), ALPHA, grid_size=800)
    r_oracle = oracle.evaluate('pit', 'Split')
    r_plain = plain.evaluate('abs_residual', 'Split')
    assert r_oracle['oracle_cond_cov_error'] < r_plain['oracle_cond_cov_error']


def test_density_scores_produce_non_convex_sets():
    """Unions of intervals are the thing an interval-only evaluation cannot represent; the grid
    measure has to actually find them."""
    dm = TimeSeriesDataModule(make_rc('regime_switch'))
    ev = TSEvaluator(dm, ts_models['RFKDE'](), ALPHA, grid_size=1500)
    hpd = ev.evaluate('hpd', 'Split')
    abs_res = ev.evaluate('abs_residual', 'Split')
    assert hpd['n_components'] > 1.0
    assert abs_res['n_components'] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------

def test_set_geometry_measures_a_union_of_two_intervals():
    grid = np.linspace(0, 10, 1001)
    membership = ((grid >= 1) & (grid <= 2)) | ((grid >= 5) & (grid <= 7))
    geom = tsm.set_geometry(membership[None, :], grid)
    assert geom['n_components'][0] == 2
    assert geom['width'][0] == pytest.approx(3.0, abs=0.05)
    assert geom['lower'][0] == pytest.approx(1.0, abs=0.02)
    assert geom['upper'][0] == pytest.approx(7.0, abs=0.02)


def test_miscoverage_runs_detects_clustering():
    clustered = np.array([True] * 90 + [False] * 10 + [True] * 900)
    spread = np.ones(1000, dtype=bool)
    spread[::100] = False
    assert tsm.miscoverage_runs(clustered)['longest_miss_run'] == 10
    assert tsm.miscoverage_runs(spread)['longest_miss_run'] == 1
    assert tsm.miscoverage_runs(clustered)['runs_test_p'] < 0.05


def test_winkler_penalises_misses():
    y = np.array([0.0, 5.0])
    lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    w = tsm.winkler_score(y, lower, upper, 0.1)
    assert w[0] == pytest.approx(2.0)
    assert w[1] > w[0]


def test_dependence_diagnostics_separate_iid_from_autocorrelated():
    rng = np.random.default_rng(7)
    iid = rng.normal(size=3000)
    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    assert abs(tsm.dependence_diagnostics(iid)['score_acf1']) < 0.1
    assert tsm.dependence_diagnostics(ar)['score_acf1'] > 0.8
    # The fitted decay should recover the true AR coefficient...
    assert tsm.dependence_diagnostics(ar)['score_acf_decay'] == pytest.approx(0.9, abs=0.06)
    # ...and must not be reported at all when there is no significant autocorrelation to fit.
    assert np.isnan(tsm.dependence_diagnostics(iid)['score_acf_decay'])


# ------------------------------------------------------------------------------------------------
# Score / model compatibility
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('model_name', ['Ridge', 'LGBMQuantile', 'GaussianRidge', 'RFKDE',
                                        'EnbPIEnsemble'])
def test_every_compatible_score_runs_for_every_model(dm, model_name):
    model = ts_models[model_name]()
    ev = TSEvaluator(dm, model, ALPHA, grid_size=400)
    for score_name in SCORES_BY_OUTPUT_TYPE[model.output_type()]:
        r = ev.evaluate(score_name, 'Split')
        assert 0.5 < r['coverage'] <= 1.0
        assert np.isfinite(r['width'])


def test_incompatible_score_is_refused(dm):
    ev = TSEvaluator(dm, ts_models['Ridge'](), ALPHA, grid_size=200)
    with pytest.raises(ValueError, match='richer model'):
        ev.evaluate('nll', 'Split')


def test_score_pass_is_cached_across_schemes(dm):
    """The decoupling that makes the cross-product affordable: adding a scheme must not recompute
    the score stream."""
    ev = TSEvaluator(dm, ts_models['Ridge'](), ALPHA, grid_size=400)
    first = ev.evaluate('abs_residual', 'Split')
    second = ev.evaluate('abs_residual', 'ACI')
    assert len(ev._score_cache) == 1
    assert second['score_time'] == first['score_time']
