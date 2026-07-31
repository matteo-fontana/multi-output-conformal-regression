# Plan: repurposing this framework as a testbed for univariate time-series conformal prediction

Target: an empirical companion to a review paper on conformal prediction for time series, taken as
a successor to Stocker, Małgorzewicz, Fontana & Ben Taieb, *A Gentle Introduction to Conformal Time
Series Forecasting* (arXiv 2511.13608). Its taxonomy, notation and validity–efficiency–compute
framing are treated as fixed inputs here rather than choices to be re-made; see §7.1 of the
companion survey.

> **Status.** Phases 0–4 of §7 are implemented and tested (`python -m pytest
> tests/test_ts_conformal.py`). See the *Time-series testbed* section of the README for usage.
> Deviations from this plan that survived contact with the code are recorded in §10.

Companion document: **`TIMESERIES_METHODS_AND_DATASETS.md`** — survey of the method families, the
datasets each paper benchmarks on (verified against the authors' released code), the proposed
dataset roster, and how this study positions against the two existing 2025–26 reviews/benchmarks.

The guiding idea is to keep the paper's organising principle — **a large factorial study over
(base predictor) × (conformity score) × (calibration scheme) × (dataset), repeated, ranked and
CD-diagrammed** — and change only what the i.i.d. assumption forces us to change.

---

## 0. What carries over, what breaks

### Carries over essentially unchanged

| Component | Why it still works |
|---|---|
| `moc/runner.py` (`Runner`, dask/joblib fan-out, `PrecomputationLevel` caching, pickled `RunConfig`s) | Task graph is agnostic to what a "run" is. |
| `moc/utils/hparams.py` (`HP` / `Join` / `Union`) + `moc/models/tuning.py` | The grid-declaration DSL is exactly the right shape for a review paper's factorial design. |
| `moc/utils/run_config.py` (`RunConfig`, `storage_path`, `hparams_str`) | Path/serialisation layout is reusable verbatim. |
| `moc/analysis/*` — `dataframes.py`, `plot.py`, `plot_cd_diagram.py` | Critical-difference diagrams over (method, dataset) mean ranks are *the* deliverable format for a review. Only the metric list changes. |
| `moc/conformal/base_conformalizer.py::conformal_quantile` | Still the split-conformal primitive; gets a weighted sibling. |
| `moc/metrics/conditional_coverage_metrics.py::wsc_unbiased` | Worst-slab coverage over lag/exogenous features is a legitimate conditional-coverage probe for time series. |
| `moc/models/quantile/quantile_model.py`, `moc/models/mixture/mixture_model.py` | Both already work at `output_dim=1`; they become the CQR and MDN base predictors for free. |
| `moc/conformal/conformalizers.py::CopulaCPTS` | Already a *time-series* method (multi-step joint bands). Free when the horizon axis is switched on. |
| `moc/metrics/cache.py` | Same trick, new keys (see §3.2). |

### Breaks and must be replaced

| Component | Problem | Fix |
|---|---|---|
| `base_datamodule.py::load_datasets` → `random_split` | Destroys temporal order; calibration set leaks the future. | Contiguous chronological splits (§2.1). |
| `base_datamodule.py::subsample` | `rng.choice` over row indices — same problem. | Contiguous truncation to the most recent `max_size` points. |
| `ConformalizerBase.get_q(alpha)` returning **one** scalar for all test points | Every adaptive method needs `q_t`. | New `OnlineConformalizer` interface (§3.1). |
| `metrics_computer.py::compute_cum_log_region_size` (Monte-Carlo region volume) | Wasteful and noisy in 1-D. | Exact Lebesgue measure on a fine grid (§4.1). |
| `MetricsComputer` returning `nan` region size when `output_type() == 'quantile'` | Quantile models are first-class citizens here. | Grid-based width works for every `output_type`. |
| `repeat_tuning` = 10 random re-splits | You cannot randomly re-split a single fixed series. | `run_id` indexes a **rolling-origin fold** (§2.3). |
| `Cache` built once from a fixed model | Invalid under rolling refit. | Cache keyed by refit epoch (§3.2). |
| Early stopping on a randomly-split val set | Leaks future into model selection. | Val block is the chronological block immediately before calibration. |

---

## 1. Repository layout

Extend in place rather than fork — the analysis stack is the expensive part and is fully shared.

```
moc/
  configs/
    ts_datasets.py           NEW  dataset groups, horizons, seasonality, series metadata
    general.py               EDIT add ts.* config block (§6)
  datamodules/
    timeseries_datamodule.py NEW  windowing, chronological split, rolling origin
    ts_synthetic.py          NEW  AR/GARCH/regime-switching generators with oracle laws
  models/
    ts/
      naive.py               NEW  naive, seasonal naive, drift
      statsmodels_model.py   NEW  ARIMA / SARIMAX / ETS wrappers
      gbm.py                 NEW  LightGBM point + pinball objectives on lag features
      rnn.py                 NEW  LSTM/GRU heads: point / quantile / mixture
      ts_oracle.py           NEW  exact conditional law for synthetic data
    __init__.py              EDIT register models + trainers
    tuning.py                EDIT add get_timeseries_tuning()
  conformal/
    scores.py                NEW  ScoreFunction family (§3.3)
    online/
      base.py                NEW  OnlineConformalizer, ThresholdUpdater
      split.py               NEW  static, sliding-window, weighted/decayed (NexCP)
      aci.py                 NEW  ACI, FACI/DtACI, AgACI
      ogd.py                 NEW  SF-OGD, SAOCP
      pid.py                 NEW  Conformal PID (+ scorecaster)
      ensemble.py            NEW  EnbPI, EnCQR
      residual_model.py      NEW  SPCI (quantile-RF on residual history)
      mvp.py                 NEW  MVP / multivalid, group-conditional
    __init__.py              EDIT registry
  metrics/
    ts_metrics.py            NEW  rolling coverage, Winkler, run-length, regret (§4)
    online_evaluator.py      NEW  sequential evaluation loop (§3.4)
  analysis/
    helpers.py               EDIT metric names/labels
    plot_timeseries.py       NEW  coverage-over-time ribbons, width trajectories
run_ts.py                    NEW  entry point mirroring run.py
```

`run.py` and the ICML pipeline stay untouched and reproducible.

---

## 2. Data layer

### 2.1 `TimeSeriesDataModule`

Subclass `BaseDataModule`, override `load_datasets` (do **not** reuse the parent's `random_split`):

1. `get_series()` returns `(y[T], exog[T, k], timestamps[T])` for one series.
2. Window into supervised form:
   `x_t = [y_{t-1..t-L}, exog_t, exog_{t-1..t-L'}, calendar(t)]`, `y_t ∈ R^h`.
   With `h = 1` this is univariate one-step-ahead; `h > 1` reuses the whole existing
   multi-output machinery (including `CopulaCPTS`, `M-CP`, `C-HDR`) for joint bands — a
   nearly-free extra section for the review.
3. Chronological contiguous split with a **gap** of `L + h - 1` between blocks to kill
   window overlap leakage:
   `train | gap | val | gap | calib | gap | test`.
   Default ratio `(0.5, 0.1, 0.2, 0.2)`; keep the existing "cap calibration at 2048" logic but
   take the **most recent** 2048 points, not a random subset.
4. Scalers fit on train only (already the case in `StandardScaler`), plus optional
   `difference` / `log` / `detrend` transforms declared per dataset in `ts_datasets.py`.
   Store the inverse transform so widths can be reported in original units.
5. `test_dataloader()` keeps `shuffle=False` (the `SequentialSampler` assertion in
   `MetricsComputer.__init__` becomes load-bearing — keep it).

### 2.2 Datasets

**Synthetic (with oracle):** these give ground-truth conditional quantiles, so you can report the
*exact* conditional-coverage error rather than a proxy. That is the single highest-value addition
for a review paper.

- AR(1)/AR(3), homoscedastic Gaussian — sanity check, all methods should tie.
- AR(1) + GARCH(1,1) noise — separates volatility-adaptive scores from plain absolute residuals.
- Regime-switching / abrupt changepoint in mean and in variance — separates ACI-family from split CP.
- Slow distribution drift (time-varying coefficients) — separates decay-weighted from static.
- Seasonal + trend + heteroscedasticity.
- Heavy-tailed (Student-t) innovations — stresses width metrics.
- Non-stationary "adversarial" sequence à la Gibbs–Candès — worst case for exchangeability.

**Real:** take the roster from the literature rather than inventing one, so every method can be
compared on its own published benchmark as well as on neutral ground. The verified per-paper
dataset inventory, the proposed groups, and acquisition/licensing notes are in
**`TIMESERIES_METHODS_AND_DATASETS.md`**. Summary: eight groups named after first authors in this
repo's existing convention — `elec2`, `xu` (EnbPI's solar/wind/appliances/greenhouse/Beijing air),
`angelopoulos` (Conformal PID's climate/stocks/COVID-deaths), `zaffran` (AgACI's AR simulations),
`auer` (HopCPT's air-quality and solar), `bhatnagar` (M4 + NN5), `gibbs` (volatility), `sun`
(CopulaCPTS, horizon > 1 only) — plus `ts_synthetic`. ELEC2 and NSRDB solar are the anchors: they
recur across otherwise disjoint communities, so nearly every method has a published number on one
of them.

Storage: `data/timeseries/<group>/<name>.csv` with a `meta.yaml` per series recording frequency,
seasonal period, target and exogenous columns, transform, upstream URL and licence. Most of the
data is vendorable directly from the authors' MIT-licensed repos — see §5 of the survey.

Keep the group→list structure of `moc/configs/datasets.py` (`get_dataset_groups`) verbatim, with
new keys `ts_synthetic`, `ts_real`, `ts_filtered`, `ts_test`.

**Adopt the field's conventions** so numbers are comparable without rescaling: `alpha = 0.1` (not
this repo's current 0.2), an explicit burn-in excluded from all metrics, and seasonal periods per
dataset (48 for ELEC2, 24 for hourly solar/air, 5 for daily stocks, 7 for daily climate).

### 2.3 Replication: rolling origin instead of random re-splits

`run_id ∈ [0, repeat_tuning)` selects fold *k* of a rolling-origin (expanding or sliding) scheme:
fold *k* ends the test block at `T_0 + k·Δ`. This restores the variance estimates the CD diagrams
need without ever training on the future. For synthetic data `run_id` additionally seeds the
generator, which is the cleaner replication.

---

## 3. Conformal layer — the core redesign

### 3.1 The key abstraction change

The current interface is static: `is_in_region(x, y, alpha)` with a single `get_q(alpha)`.
Time-series methods need a *stream*. Introduce:

```python
class OnlineConformalizer:
    def q_t(self, t): ...                 # threshold used at test index t
    def observe(self, t, score_t): ...    # feedback after y_t is revealed
    def is_in_region(self, t, x, y, alpha): ...
```

with the static split conformalizer becoming the degenerate case (`q_t` constant, `observe` a
no-op). This keeps every existing method usable as a baseline.

### 3.2 The decoupling that makes this cheap

Almost every online method updates only a *threshold*, not the model. So split the pipeline:

1. **Score pass (vectorised, model-heavy, cacheable).** With a fixed fitted model, compute
   `s_t` for all calibration and test indices in batched GPU passes — exactly the role
   `moc/metrics/cache.py` plays today. Extend `Cache` with keys `scores`, `quantiles`,
   `residuals`, `sigma_hat`, and key the whole cache by *refit epoch* so rolling refit
   invalidates correctly.
2. **Threshold pass (sequential, pure numpy, microseconds).** ACI / DtACI / PID / SF-OGD /
   SAOCP / NexCP / sliding-window all consume the precomputed score stream and emit `q_t` in an
   O(T) loop with zero model calls.
3. **Model-in-the-loop methods** (SPCI, EnbPI, HopCPT, any rolling-refit configuration) declare
   `requires_sequential_model_calls = True` and go through the slow path. Gate them behind a
   config flag so a full sweep stays affordable.

This mirrors the paper's own "generalized conformity score" decomposition and is the main reason
the existing architecture is a good fit.

### 3.3 Score families (`conformal/scores.py`)

| Score | `output_type` needed | Notes |
|---|---|---|
| Absolute residual `\|y − ŷ\|` | point | baseline |
| Signed residual (two one-sided calibrations) | point | asymmetric intervals |
| Normalised residual `\|y − ŷ\| / σ̂_t` | point + variance head | `σ̂_t` from a GARCH fit, a rolling std, or a learned head |
| CQR `max(q̂_lo − y, y − q̂_hi)` | quantile | Romano et al. |
| CQR-r / normalised CQR | quantile | Sesia–Candès |
| NLL / DR-CP `−log p̂(y\|x)` | distribution | already in repo |
| HPD / C-HDR | distribution | already in repo; yields non-convex sets in 1-D |
| PIT / rank score `\|F̂(y\|x) − 1/2\|` | distribution | equal-tailed by construction |
| PCP / HD-PCP sample-based | distribution | already in repo |

The first eight columns of the results table are then directly comparable to the multi-output
paper's Table 1 in structure.

### 3.4 Calibration schemes (`conformal/online/`)

| Scheme | Reference |
|---|---|
| Split CP (static) | Vovk / Papadopoulos — the "does nothing" control |
| Sliding-window CP | folklore baseline |
| Weighted / exponentially-decayed CP (NexCP) | Barber, Candès, Ramdas, Tibshirani 2023 |
| Blocked / randomisation CP | Chernozhukov, Wüthrich, Zhu 2018 |
| ACI | Gibbs & Candès 2021 |
| FACI / DtACI | Gibbs & Candès 2024 |
| AgACI | Zaffran et al. 2022 |
| SF-OGD, SAOCP | Bhatnagar et al. 2023 |
| Conformal PID (+ scorecaster) | Angelopoulos, Candès, Tibshirani 2023 |
| MVP / multivalid | Bastani et al. 2022 |
| EnbPI, EnCQR | Xu & Xie 2021; Jensen et al. 2022 |
| SPCI | Xu & Xie 2023 |
| HopCPT (optional, heavier) | Auer et al. 2023 |
| CF-RNN (Bonferroni), CopulaCPTS | Stankevičiūtė et al. 2021; Sun & Yu 2024 — horizon axis only |
| TQA / CPTD | Lin, Trivedi & Sun 2022 — panel/longitudinal coverage |

See `TIMESERIES_METHODS_AND_DATASETS.md` §1 for the full taxonomy, what each method actually
adapts (calibration weights vs. threshold vs. residual law vs. horizon), and links to the
reference implementations — several of which (`salesforce/online_conformal`) ship independent
implementations of split CP, NexCP and FACI that are useful for cross-checking ours.

Two shared primitives to add next to `conformal_quantile`:

```python
def weighted_conformal_quantile(scores, weights, alpha):  # NexCP: w̃_i = w_i / (Σw + 1), +∞ atom
def rolling_conformal_quantile(scores, window, alpha):    # sliding window, vectorised
```

The `α_t` recursion of the ACI family should live in one place and be shared by all its variants,
so the comparison isolates the learning rate / aggregation rule rather than implementation noise.

---

## 4. Metrics (`metrics/ts_metrics.py`)

Marginal coverage alone is nearly uninformative for time series — every method here is designed to
hit it. The interesting axes are *when* the misses happen and *what they cost*.

### 4.1 Set size, exactly

In 1-D there is no need for the Monte-Carlo volume estimator. Evaluate membership on a fine grid
over `[min − c·range, max + c·range]` and report the Lebesgue measure directly. This is exact up to
grid resolution, cheap, and handles the non-convex sets produced by HPD/PCP-style scores — which
convex-interval-only evaluations in the literature typically cannot. Also report the number of
connected components, a genuinely new descriptive statistic for a review.

### 4.2 Metric list

Grouped under the **validity / efficiency / compute** triad used in *A Gentle Introduction to
Conformal Time Series Forecasting*, so the results tables compose with that paper's rather than
competing with them.

**Validity**

- **Marginal coverage** and its deviation from `1 − α`.
- **Rolling / local coverage**: mean coverage in sliding windows of width `W ∈ {50, 100, 250}`;
  report RMS deviation ("LCE") and worst-window deviation.
- **Miscoverage clustering**: longest run of consecutive misses, run-length distribution, and a
  runs test / Ljung–Box on the miscoverage indicator sequence. Independence of the error sequence
  is the property split CP loses first, and it is the metric most directly tied to the
  weak-dependence conditions in the prior review's theory section.
- **Conditional coverage**: `wsc_unbiased` on lag features (reuse as-is); coverage stratified by
  volatility decile and by regime label; and on synthetic data the **exact** conditional-coverage
  error against the oracle law.
- **Shift response**: coverage in windows before/at/after each known changepoint, and recovery time
  (steps until rolling coverage re-enters a band around `1 − α`).
- **Dependence diagnostics** (new, and the bridge to the theory): estimated mixing / weak-dependence
  coefficients per dataset, and the realised coverage loss plotted against the bound they imply.
  On the synthetic group these are known by construction, so the bound's tightness is measurable
  rather than merely assertable — see §7.2 of the survey.

**Efficiency**

- **Set size**: mean, median, geometric mean (reuse the existing log-size plumbing), plus number
  of connected components.
- **Winkler / interval score** and **pinball loss** at `α/2, 1 − α/2` — proper scoring, ties the
  study to the forecasting literature.
- **Adaptivity**: Spearman correlation between width `t` and `|residual_t|` (or oracle conditional
  σ_t where available).
- **Regret** vs. the best fixed threshold in hindsight, and strongly-adaptive regret over intervals
  — the natural yardstick for the online-learning methods.

**Compute**

- **Cost**: keep `score_time` / `test_coverage_time` / `total_time`, and add
  `update_time_per_step`, which is what actually matters for deployment.

Add all names to `moc/analysis/helpers.py::main_metrics` / `other_metrics` and
`get_metric_name`; the CD-diagram and table code then picks them up with no further changes.

---

## 5. Base predictors

`output_type()` gains a `'point'` member alongside `'distribution'` and `'quantile'`; the
grid-based size computation in §4.1 makes `MetricsComputer` work uniformly across all three.

Recommended roster, spanning the cost/quality range a review should cover:

- **Naive, seasonal naive, drift** — cost floor; also isolates how much of a method's coverage
  comes from calibration rather than the model.
- **Ridge / linear on lags** — classical.
- **ARIMA/SARIMAX, ETS** (statsmodels) — the statistical baseline, and the only ones giving
  model-based predictive variances to contrast with conformal ones.
- **LightGBM** on lag features, point and pinball objectives.
- **Quantile MLP** — `QuantileModule` already works at `output_dim=1`.
- **Mixture density network** — `MixtureLightningModule` already works at `output_dim=1`.
- **LSTM/GRU** with point / quantile / mixture heads — the standard deep baseline in this literature.
- **MQF2** — already present, and originally a *time-series* quantile-function forecaster; keeps
  the latent-space methods (`L-CP`, `L-H`, `STDQR`) in play.
- **Oracle** — synthetic data only; upper bound on achievable conditional coverage.

**Refit protocol** as an explicit hyperparameter, not a hidden choice: `fit_once` /
`refit_every_k` / `expanding_window` / `sliding_window`. Several published comparisons differ
mainly here, and making it a first-class axis is itself a review-paper contribution.

---

## 6. Config and entry point

Add to `general_config`:

```python
horizon=1,
lags=24,
seasonality=None,          # per-dataset override
gap_between_splits=True,
refit_policy='fit_once',   # fit_once | refit_every_k | expanding | sliding
refit_every=100,
rolling_origin_step=None,  # defaults to test-block length
alpha=0.1,                 # 0.2 in the ICML setup; 0.1 is the TS convention
grid_size=2000,            # for exact 1-D set measure
local_coverage_windows=(50, 100, 250),
allow_sequential_model_methods=False,
```

`run_ts.py` mirrors `run.py` with `tuning_type='timeseries'`. Then:

```bash
python run_ts.py name="ts_full" device="cuda" datasets="ts_filtered" repeat_tuning=10
```

`get_timeseries_tuning()` in `models/tuning.py` expresses the factorial design directly in the
existing DSL:

```python
Join(
    HP(model=['SeasonalNaive', 'ARIMA', 'LightGBM', 'QuantileLSTM', 'MixtureLSTM']),
    HP(posthoc_grid=Union(
        Join(HP(method='Split'),   HP(score=all_scores)),
        Join(HP(method='Rolling'), HP(score=all_scores), HP(window=[100, 500, 2000])),
        Join(HP(method='NexCP'),   HP(score=all_scores), HP(rho=[0.99, 0.995, 0.999])),
        Join(HP(method='ACI'),     HP(score=all_scores), HP(gamma=[0.001, 0.005, 0.01, 0.05])),
        Join(HP(method='DtACI'),   HP(score=all_scores)),
        Join(HP(method='PID'),     HP(score=all_scores), HP(kp=[...], ki=[...])),
        Join(HP(method='SAOCP'),   HP(score=all_scores)),
    )),
)
```

Note the shape: **score × scheme is a full cross-product**, which is precisely the "unified
comparison" framing of the original paper, transplanted. Most published TS papers evaluate a
single score with a single scheme; filling in this grid is the empirical contribution.

---

## 7. Phasing

**Phase 0 — data acquisition (~0.5 day).** `scripts/fetch_ts_data.py`: shallow-clone the source
repos, copy the vendorable series into `data/timeseries/<group>/`, write `meta.yaml` per series,
and record upstream commit SHAs (several of these repos have re-uploaded their data files). This
is cheap and unblocks everything else; see §5 of the survey.

**Phase 1 — minimum viable testbed (~2–3 days).** `TimeSeriesDataModule` with chronological
splits + windowing; 3 synthetic + 2 real datasets; naive/ridge/LightGBM models; absolute-residual
and CQR scores; Split + Rolling + ACI schemes; coverage / rolling coverage / mean width metrics;
`run_ts.py`. End state: `python run_ts.py datasets=ts_test fast=true` produces a dataframe the
existing `analysis.ipynb` machinery can read.

**Phase 2 — score families (~2 days).** `conformal/scores.py` with all nine scores; wire the
distribution-based ones to the existing MDN/MQF2 models; grid-based exact set measure with
component counting.

**Phase 3 — scheme families (~3–4 days).** DtACI, AgACI, SF-OGD, SAOCP, PID, NexCP, MVP, EnbPI;
`weighted_conformal_quantile` and the shared `α_t` recursion.

**Phase 4 — metric depth (~2 days).** Winkler, pinball, run-length/independence tests, adaptivity
correlation, regret, changepoint response, oracle conditional coverage on synthetic data.

**Phase 4b — reproduction gate (~1 day).** Before generating any novel number, reproduce two
published results: the SPCI-vs-EnbPI interval-width reduction on ELEC2 (the SPCI repo ships this
exact comparison as a notebook) and the AgACI AR(1) φ = 0.9 learning-rate sweep. Both are cheap
and both have numbers to check against. See §4 of the survey for the full reproduction matrix.

**Phase 5 — scale-out and analysis (~2–3 days).** Full dataset roster, rolling-origin replication,
CD diagrams, per-dataset tables, coverage-over-time figures. Reuse `plot_cd_diagram.py` unchanged.

**Phase 6 — optional extensions.** Model-in-the-loop methods (SPCI, HopCPT, rolling refit);
multi-step horizon `h > 1` reusing `CopulaCPTS` / `M-CP` / `C-HDR` for joint bands.

Phases 1–5 give a complete, publishable empirical section; Phase 6 is upside.

---

## 8. Correctness traps to guard with tests

1. **No future leakage.** Assert `max(train_idx) < min(val_idx) < min(calib_idx) < min(test_idx)`
   and that the gap ≥ `L + h − 1`. A unit test that shuffles the series and checks the pipeline
   *fails* to reach nominal coverage is a good canary.
2. **Scalers fit on train only** — already true, but re-assert after the split rewrite.
3. **`conformal_quantile` weight normalisation.** The `+∞` atom in the NexCP formulation is easy
   to drop; test against the unweighted case with uniform weights.
4. **Off-by-one in the online loop.** `q_t` must be computed *before* `y_t` is observed. Test:
   feeding a method the true `y_t` early should visibly inflate coverage.
5. **Cache invalidation under refit.** Test that changing `refit_policy` changes the cached scores.
6. **`α_t` clipping.** ACI's `α_t` can leave `[0, 1]`; the convention (infinite or empty set) must
   be explicit and counted, since it silently distorts mean width.
7. **Grid coverage.** Assert the evaluation grid contains the realised `y_t`, otherwise widths are
   truncated and coverage is understated.

---

## 9. Why this framework specifically

Three properties of the existing code make it a better starting point than writing fresh:

1. The **score/method separation** already present in `base_conformalizer.py` maps almost exactly
   onto the score/scheme separation the time-series literature needs and rarely makes explicit.
2. The **caching layer** (`metrics/cache.py`) is what makes a full cross-product affordable; the
   §3.2 decoupling extends it rather than replacing it.
3. The **analysis stack** — pickled `RunConfig`s → tidy dataframe → CD diagrams and LaTeX tables —
   is the part that is tedious to rebuild and needs essentially no changes.

The honest caveat: this framework's central assumption is exchangeability, and the whole point of
the new study is that it fails. Everything in §0's "breaks" table is a place where that assumption
is baked into code that looks generic. The rewrite is genuinely mechanical, but it must be done
deliberately rather than by inheriting from `BaseDataModule` and hoping.

---

## 10. What was actually built, and where it departs from this plan

Phases 0–4 are implemented; 53 tests pass. This section records the deviations, so the plan and
the code do not drift apart silently.

### Design changes made during implementation

1. **`TimeSeriesDataModule` does not subclass `BaseDataModule`.** §2.1 proposed subclassing. In
   practice the parent's substance is `random_split` plus DataLoader plumbing — the first is
   exactly what must not happen here, and the second buys nothing for a conformal core that is
   vectorised numpy. The datamodule is therefore standalone and torch-free. What is reused from
   the architecture is everything *above* it: `RunConfig`, the runner, the `HP`/`Join`/`Union`
   DSL, and the analysis stack.
2. **A separate `TSEvaluator` rather than an extended `MetricsComputer`.** §4 assumed the existing
   metrics computer could be made to work for all three `output_type`s. It was cleaner to write a
   dedicated driver: the multi-output one is organised around Monte-Carlo region volume and a
   fixed `q`, and the time-series one needs an exact 1-D grid measure and a `q_t` stream.
3. **The score/threshold decoupling of §3.2 is the load-bearing decision** and survived intact.
   Adding a scheme to a sweep costs milliseconds because the score pass is cached per
   `(model, score)`; a test asserts this (`test_score_pass_is_cached_across_schemes`).
4. **Two upstream files were changed** to keep the time-series path from requiring the multi-output
   dependency stack: `moc/models/__init__.py` and `moc/datamodules/__init__.py` now resolve their
   registries lazily, so `cpflows`, `rpy2`/`drf` and `torchvision` are needed only by the models
   that use them. Behaviour for existing callers is unchanged.

### Implemented

- 11 synthetic generators with exact conditional laws, and 25 vendored real series across four
  groups (`scripts/fetch_ts_data.py`).
- 8 scores, 11 schemes, 9 base models, and the full compatibility-filtered cross-product.
- The metric suite of §4 under validity / efficiency / compute, including exact set measure with
  connected-component counts, miscoverage run-length and runs test, adaptivity correlations,
  regret, changepoint response, and oracle conditional coverage.
- Score-stream dependence diagnostics (ACF, fitted geometric decay, Ljung–Box).

### Not implemented, and why

- **Horizon > 1.** `make_windows` raises rather than silently doing something wrong. This is
  Phase 6, and it is where `CopulaCPTS`, `CF-RNN` and the `sun` dataset group come in.
- **Refit protocols.** Only `fit_once` exists; `refit_every_k` / `expanding` / `sliding` are
  declared in §5 as a first-class axis but not yet wired. The cache would need keying by refit
  epoch, as §3.2 anticipates.
- **ARIMA/ETS, LSTM/GRU, and reuse of the repo's `QuantileModule` / `MixtureLightningModule` /
  `MQF2`.** The model roster covers point, quantile and distribution output types with
  numpy/sklearn/LightGBM predictors, which is enough to exercise every score. Adding the deep and
  statistical baselines is additive: implement `fit`/`predict`/`output_type` and register.
- **AgACI, MVP, EnCQR, HopCPT, blocked/randomisation CP.** The scheme registry covers the ACI,
  OGD/coin-betting, PID, weighted and ensemble families; these five are the remaining entries from
  the survey's §1.
- **The `gibbs`, `auer`, `bhatnagar` and `sun` dataset groups.** Registered in the survey but not
  vendored: NSRDB beyond the EnbPI extracts needs an API key, LamaH-CE streamflow needs
  `neuralhydrology`, M4/NN5 come through Merlion, and the CopulaCPTS series are multi-horizon.
- **The weak-dependence bound comparison itself.** The diagnostics that feed it are in place
  (`dependence_diagnostics`, `true_mixing_rate`), but computing the finite-sample bound from the
  prior review and plotting realised coverage loss against it is the next piece of analysis work,
  not testbed plumbing.

### Reproduction gate (§7, Phase 4b) — run, both checks pass

`python scripts/reproduce_published.py` checks the testbed against two published results before
its novel cells are trusted.

**SPCI vs EnbPI on ELEC2** (Xu & Xie 2023). Base model is the bootstrap ensemble in both arms, so
the only difference is the calibration scheme:

| method | coverage | width | Winkler | LCE(100) | update time |
|---|---|---|---|---|---|
| EnbPI (ensemble + `Rolling`) | 0.899 | 0.0578 | 0.0783 | 0.031 | 0.01 s |
| SPCI (ensemble + quantile forest) | 0.903 | 0.0539 | 0.0682 | 0.029 | 17.0 s |
| Split (control) | 0.935 | 0.0672 | 0.0790 | 0.046 | 0.00 s |

SPCI is **6.9% narrower at matched (slightly higher) coverage**, reproducing the published
direction, and pays for it with three orders of magnitude more update time — the cost axis the
original comparison does not report. Static split conformal over-covers by 3.5 points, which is the
control behaving as the theory says it should.

**ACI's learning-rate sensitivity** on AgACI's AR(1) with φ = 0.9 (Zaffran et al. 2022). Sweeping
γ over [0.0005, 0.5] moves the mean width by **241% of its mean** while marginal coverage stays
pinned near 0.89–0.90 — precisely the "coverage is not the problem, γ is" motivation for
aggregating over learning rates. DtACI lands at 3.29, the narrow end of the range, without being
told the right γ.

Both checks assert their published *direction* rather than exact numbers, since the base
predictors and splits differ from the originals.

### A first finding from the cross-product

The score axis immediately surfaces a confound that a score-fixed benchmark cannot see: **the
OGD/PID learning rate has to be chosen per score family, not per dataset.**

`RFKDE` on `ar1_garch`, α = 0.1, `PID` at three learning rates:

| score | range | lr | coverage | width | grid-truncated |
|---|---|---|---|---|---|
| `hpd` | [0, 1] | 0.001 | 0.892 | 3.80 | 0% |
| `hpd` | [0, 1] | 0.01 | 0.896 | 4.26 | 2.5% |
| `hpd` | [0, 1] | 0.10 | 0.899 | 8.69 | 27.4% |
| `abs_residual` | [0, 4.0] | 0.001 | 0.874 | 3.28 | 0% |
| `abs_residual` | [0, 4.0] | 0.01 | 0.883 | 3.36 | 0% |
| `abs_residual` | [0, 4.0] | 0.10 | 0.898 | 3.50 | 0% |

`lr = 0.1` is the published default and is well-scaled for a residual score, whose range is set by
the data. On a *bounded* score it is a step of 10% of the entire score domain, so the threshold
routinely exceeds the score's own maximum, the prediction set becomes the whole line, and the mean
width more than doubles — at a marginal coverage that still looks correct. Only the width and the
`truncated_rate` diagnostic reveal it.

Two consequences: the tuning grid now spans `lr ∈ [0.001, 0.5]` so the sweep can find the right
scale per score, and `truncated_rate` earns its place in the metric table (guard 7 of §8 paying
for itself). More generally this is the kind of result the study exists to produce — an
interaction between the two axes that neither axis alone can expose.
