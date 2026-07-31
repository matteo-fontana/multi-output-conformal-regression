# Survey: conformal prediction for time series — methods and their benchmark datasets

Companion to `TIMESERIES_TESTBED_PLAN.md`. The purpose is to fix the testbed's dataset roster to
the benchmarks the literature actually uses, so that every method in the study can be compared
*on its own home turf* as well as on neutral ground.

**Provenance markers.** Facts below are marked:
- **[repo]** — verified by inspecting the authors' released code/data (shallow clone, file
  contents, config files). Highest confidence; row counts and column names are literal.
- **[web]** — from search summaries of the paper. Reliable for "which dataset", less so for exact
  spans; re-check against the PDF before the dataset table goes in the paper.

Note on this environment: `arxiv.org`, `openreview.net` and `proceedings.mlr.press` are blocked by
the sandbox network policy, `github.com` is not. Everything marked **[repo]** was therefore checked
directly; everything marked **[web]** could not be.

---

## 1. Method taxonomy

Six families. The axis that matters for the testbed is *what the method adapts* — the base model,
the score, the calibration set, or the threshold — because that determines where it plugs into the
architecture in §3 of the plan.

### F1. Static split CP applied naively (the control)

Split conformal with a single `q` from a held-out calibration block. Included as the baseline that
*should* fail: its coverage degrades exactly as the score sequence departs from exchangeability, and
quantifying that degradation per dataset is a useful review-paper result in itself.

- Vovk et al. 1999/2005; Lei et al. 2018 (split CP).
- **Oliveira et al. 2024, "Split conformal prediction and non-exchangeable data", JMLR 25** — the
  reference for *how much* coverage you lose; gives bounds in terms of mixing coefficients. **[web]**

### F2. Reweighted / non-exchangeable calibration — *adapts the calibration weights*

Keeps the split-CP structure but downweights stale calibration points.

- **NexCP — Barber, Candès, Ramdas & Tibshirani 2023, "Conformal prediction beyond
  exchangeability", Annals of Statistics 51(2)**. Fixed weights `w_i` (typically `ρ^{n+1-i}`),
  normalised as `w̃_i = w_i / (Σ_j w_j + 1)` with a `+∞` atom; coverage loss bounded by the total
  variation between the data sequence and its swapped version. Also introduces a randomisation
  trick for non-symmetric algorithms. **[web]**
- Tibshirani, Barber, Candès & Ramdas 2019 (weighted CP under covariate shift) — the antecedent.

### F3. Online threshold adaptation — *adapts `q_t` from realised miscoverage*

The largest family, and the one where the testbed's §3.2 decoupling pays off: all of these are O(T)
recursions over a precomputed score stream.

- **ACI — Gibbs & Candès 2021, NeurIPS**. `α_{t+1} = α_t + γ(α − err_t)`. **[repo:
  `isgibbs/AdaptiveConformal`, a single `ACCode.R`]**
- **FACI / DtACI — Gibbs & Candès 2024, JMLR 25(162)**. Expert aggregation over a set of candidate
  `γ`, removing the learning-rate choice. **[web]**
- **AgACI — Zaffran, Féron, Goude, Josse & Dieuleveut 2022, ICML**. Online aggregation of ACI
  experts under an ARMA-style analysis; the paper is also the most careful *empirical* comparison of
  its era. **[repo: `mzaffran/AdaptiveConformalPredictionsTimeSeries`, `AgACI/` in R]**
- **SF-OGD and SAOCP — Bhatnagar, Wang, Xiong & Bai 2023, ICML**. Scale-free online gradient
  descent on the pinball loss, plus scale-free expert aggregation over dyadic intervals giving
  *strongly adaptive* regret. **[repo: `salesforce/online_conformal`; also ships reference
  implementations of split CP, NexCP and FACI, which is convenient for cross-checking ours]**
- **Conformal PID — Angelopoulos, Candès & Tibshirani 2023, NeurIPS**. Reads ACI as pure integral
  control and adds proportional (quantile tracking) and derivative-ish (a *scorecaster* that
  forecasts the score itself) terms. **[repo: `aangelopoulos/conformal-time-series`; methods in
  `core/methods.py` are `trailing_window`, `aci`, `aci_clipped`, `quantile`,
  `quantile_integrator_log`, `quantile_integrator_log_scorecaster`]**
- **MVP — Bastani, Gupta, Jung, Noarov, Ramalingam & Roth 2022, NeurIPS**. Multivalid /
  group-conditional coverage against adversarial sequences. **[web]**
- **Decaying step sizes — Angelopoulos, Barber & Bates 2024**; **Bellman Conformal Inference —
  2024**, which plans `q_t` over a horizon by dynamic programming. **[web]**

### F4. Ensemble / bootstrap residuals — *adapts the calibration set by reusing training data*

- **EnbPI — Xu & Xie 2021, ICML (oral); extended in IEEE TPAMI 2023**. Bootstrap ensemble,
  leave-one-out residuals, sliding window; no data splitting. Widely integrated downstream (MAPIE,
  AWS Fortuna, PUNCC, functime, ConformalPrediction.jl). **[repo: `hamrel-cxu/EnbPI`]**
- **EnCQR — Jensen, Bianchi & Anfinsen 2022, IEEE TNNLS**. Ensemble + CQR. **[web]**

### F5. Modelling the residual process — *adapts the score's conditional law*

The most statistically interesting family for a review: these exploit the very dependence the other
families treat as a nuisance.

- **SPCI — Xu & Xie 2023, ICML**. Fits a quantile random forest to the *past residual sequence* and
  uses its conditional quantiles as `q_t`. **[repo: `hamrel-cxu/SPCI-code`]**
- **HopCPT — Auer, Gauch, Klotz & Hochreiter 2023, NeurIPS**. A modern Hopfield network retrieves
  past time steps in similar regimes and builds a weighted residual distribution from them.
  **[repo: `ml-jku/HopCPT`]**
- **CPTC — Conformal Prediction for Time-series with Change points, 2025**. Couples a latent-state
  model with online CP. **[web; code `Rose-STL-Lab/CPTC`]**

### F6. Multi-horizon joint bands — *adapts across the horizon dimension*

Univariate in the marginal, multi-output in the horizon — which is exactly where this repository's
existing machinery already applies.

- **CF-RNN — Stankevičiūtė, Alaa & van der Schaar 2021, NeurIPS**. Bonferroni correction across the
  horizon. **[repo: `kamilest/conformal-rnn`]**
- **CopulaCPTS — Sun & Yu 2024, ICLR**. Learns a copula over per-horizon miscoverage instead of a
  union bound. **[repo: `Rose-STL-Lab/CopulaCPTS`; already implemented here as
  `moc/conformal/conformalizers.py::CopulaCPTS`]**
- **TQA — Lin, Trivedi & Sun 2022, NeurIPS** ("Conformal Prediction with Temporal Quantile
  Adjustments", arXiv 2205.09940) and the distinct **CPTD — same authors, TMLR 2022** ("Conformal
  Prediction Intervals with Temporal Dependence", arXiv 2205.12940). Both target *longitudinal*
  coverage on panels of series; CPTD proves distribution-free longitudinal validity is impossible
  and settles for cross-sectional validity plus improved longitudinal behaviour. **[web]**
- **Multi-step online CP**: several 2024–2026 entries (AcMCP, optimisation-based multi-step OCP).
  **[web]**

### Also worth a paragraph each in the review, no implementation needed

- **Chernozhukov, Wüthrich & Zhu 2018 (COLT) / 2021 (JASA)** — blocked-permutation conformal
  inference for dependent data; the honest "exact under a symmetry assumption" alternative.
- **Podkopaev & Ramdas** — label shift; adjacent but frequently cited in this literature.

---

## 2. What each paper benchmarks on

| Method | Real datasets | Synthetic | Base forecasters | Prov. |
|---|---|---|---|---|
| **ACI** (Gibbs & Candès '21) | Stock-market volatility; 2020 US election-night vote-count prediction | — | GARCH-style volatility regressions; election-night state models | [repo]+[web] |
| **NexCP** (Barber et al. '23) | ELEC2 electricity; election forecasting | Changepoint / drifting AR simulations | Least squares, RF | [web] |
| **FACI/DtACI** (Gibbs & Candès '24) | ELEC2 (predict `transfer` from NSW/VIC prices and demands); market volatility | Adversarial shift sequences | — | [web] |
| **AgACI** (Zaffran et al. '22) | French electricity spot prices, 2016–2019 (**34,896** hourly rows; columns `Spot`, `hour`, day-of-week dummies, `lag_24_*`, …; built from **eco2mix**) | ARMA(1,1)/AR(1) with φ = 0.9, Friedman-style mean function, `n=300`, `T₀=200`, 500 reps | Random forest | [repo] |
| **EnbPI** (Xu & Xie '21) | `Solar_Atl` NSRDB Atlanta hourly (**8,760**, lat 33.76/lon −84.39); `Wind_Hackberry` hourly MWH 2019–2020 (**13,871**); UCI Appliances energy 10-min (**19,735**); UCI Greenhouse-gas (**16 series × 327**); UCI Beijing multi-site air quality, Tiantan station, hourly (**35,064**); 9 Bay-Area NSRDB solar sites for 2018 (Palo Alto, Fremont, Milpitas, Mountain View, North San Jose, Redwood City, San Mateo, Santa Clara, Sunnyvale; **8,760** each) | — | RF, Ridge, NN, RNN | [repo] |
| **SPCI** (Xu & Xie '23) | `electricity-normalized.csv` = ELEC2 (**45,312** half-hourly); solar; wind | Three generators in `data.py`: state-space, non-stationary, heteroskedastic | RF, Ridge, NN | [repo] |
| **SAOCP / SF-OGD** (Bhatnagar et al. '23) | M4 Hourly / Daily / Weekly, NN5 Daily (via Merlion `ts_datasets`), horizon 24 | — | LGBM, ARIMA, Prophet | [repo] |
| **Conformal PID** (Angelopoulos et al. '23) | ELEC2 (**45,312** half-hourly, NSW, seasonal period 48); Delhi daily climate (**1,574** daily: `meantemp`, `humidity`, `wind_speed`, `meanpressure`); AMZN / GOOGL / MSFT daily open from a DJIA panel (**93,612** rows from 2006-01-03, `log: True`, seasonal period 5); COVID-19 4-week-ahead statewide death forecasting for AK/CA/FL/GA/KS/NY/TX (truncated at 106 timestamps), benchmarked against the CDC ensemble | `stationary.yaml`, `increasing_lownoise.yaml`, `mix1.yaml` | `ar`, `theta`, `prophet`, `transformer`; score `signed-residual`; α = 0.1; `T_burnin` 300 (elec2) / 100 (stocks) | [repo] |
| **HopCPT** (Auer et al. '23) | NSRDB solar 2018–20 hourly ("Solar 3Y"), NSRDB 2019 ("Solar 1Y"), EnbPI's solar set ("Solar Small"); Beijing air PM2.5 and PM10; streamflow (`hydro`, via `neuralhydrology`); sapflux | — | `darts_forest` (RF), `darts_lightgbm`, Ridge, LSTM; plus MC-dropout, MDN, Gaussian NN baselines. Compared against EnbPI, SPCI, NexCP, standard CP, CopulaCPTS, AdaptiveCI, CF-RNN | [repo] |
| **CF-RNN** (Stankevičiūtė et al. '21) | MIMIC-III (PhysioNet-credentialed); UCI EEG database (#121); COVID-19 UK (UKHSA dashboard) | Autoregressive synthetic | RNN / BJ-RNN | [repo] |
| **CopulaCPTS** (Sun & Yu '24) | NRI Particles; Drone (PythonRobotics); UK COVID epidemiology; Argoverse 1 trajectories | — | RNN encoder-decoder | [repo] |

**Convergence to note in the review:** ELEC2 and NSRDB solar are the two datasets that recur across
otherwise disjoint communities (ELEC2 in NexCP, DtACI, SPCI, Conformal PID; NSRDB solar in EnbPI,
SPCI, HopCPT). They are the natural "anchor" datasets — every method has a published number on at
least one of them.

---

## 3. Proposed testbed roster

Mirroring this repo's existing convention of naming dataset groups after the first author
(`camehl`, `cevid`, `del_barrio`, `feldman`, `mulan`, `wang`), giving eight time-series groups plus
synthetic.

| Group | Datasets | Source | Obtainable? |
|---|---|---|---|
| `ts_synthetic` | `ar1_gauss`, `ar1_garch`, `ar1_heavytail`, `changepoint_mean`, `changepoint_var`, `regime_switch`, `slow_drift`, `seasonal_hetero`, `adversarial` | Generated in `moc/datamodules/ts_synthetic.py` | Yes — and the only group with an oracle conditional law |
| `zaffran` | `ar_phi09`, `arma_friedman` | Reimplement `generation.py` from the AgACI repo | Yes |
| `xu` | `solar_atl`, `wind_hackberry`, `appliances`, `greenhouse`, `beijing_tiantan`, `solar_bayarea_{9 sites}` | Vendored in `hamrel-cxu/EnbPI/Data` (MIT) | Yes — direct copy |
| `elec2` | `elec2`, `elec2_transfer` | Vendored identically in the SPCI and Conformal PID repos | Yes — direct copy |
| `angelopoulos` | `daily_climate`, `amzn`, `googl`, `msft`, `covid_deaths_{ak,ca,fl,ga,ks,ny,tx}` | `aangelopoulos/conformal-time-series/tests/datasets` (MIT) | Mostly — `deaths.csv` for the COVID rebuild is behind a Google Drive link; the per-state `*_proc_4wkdeaths.pkl` are vendored and sufficient |
| `auer` | `beijing_pm25`, `beijing_pm10`, `nsrdb_solar_1y`, `nsrdb_solar_3y`, `streamflow`, `sapflux` | `ml-jku/HopCPT` configs; NSRDB needs an API key, streamflow needs `neuralhydrology`/LamaH-CE | Partly — start with the Beijing series (UCI, open) |
| `bhatnagar` | `m4_hourly`, `m4_daily`, `m4_weekly`, `nn5_daily` | Merlion `ts_datasets`; M4 and NN5 are open | Yes |
| `gibbs` | `etf_volatility`, `election_2020` | Volatility rebuildable from any daily OHLC source; election data is bespoke | Volatility yes; election — skip, it is not really a forecasting task |
| `sun` | `covid_uk`, `particles`, `drone` | `Rose-STL-Lab/CopulaCPTS/data` | Yes — but multi-horizon only (Phase 6) |

**Recommended `ts_filtered` default** (the analogue of the repo's existing `filtered_datasets`), a
9-dataset set that covers every mechanism at manageable cost:

```
ts_synthetic:  ar1_garch, changepoint_var, slow_drift
elec2:         elec2
xu:            solar_atl, wind_hackberry, beijing_tiantan
angelopoulos:  daily_climate, amzn
```

### Deliberately excluded

- **MIMIC-III** (CF-RNN) — PhysioNet credentialing makes results irreproducible for readers.
- **Argoverse** (CopulaCPTS) — large, and trajectory forecasting is a different problem.
- **ImageNet-C / TinyImageNet** (SAOCP) — distribution-shifted classification, not time series.
- **Election night** (ACI, NexCP) — a genuine one-off, not a repeatable benchmark.

---

## 4. Reproduction matrix

Which published headline experiment each group lets you reproduce — the evidence that the testbed's
implementations are faithful before you trust its novel cells.

| Group | Reproduces |
|---|---|
| `elec2` | NexCP Fig. on electricity; DtACI ELEC2; SPCI vs EnbPI width reduction (the SPCI repo ships this exact comparison as `tutorial_electric_EnbPI_SPCI.ipynb`); Conformal PID `elec2.yaml` |
| `xu` | EnbPI's full ICML/TPAMI table; SPCI's solar and wind results; HopCPT's "Solar Small" |
| `angelopoulos` | Conformal PID on stocks, climate and COVID-death forecasting |
| `zaffran` | AgACI's AR(1) φ = 0.9 simulation study, incl. its ACI-vs-EnbPI comparison |
| `bhatnagar` | SAOCP / SF-OGD on M4 and NN5 at horizon 24 |
| `auer` | HopCPT's Beijing air rows |
| `ts_synthetic` | Nothing published — this is where the oracle-conditional-coverage results live |

A useful sanity gate before running anything novel: reproduce the SPCI-vs-EnbPI width reduction on
`elec2` and the ACI learning-rate sweep from `zaffran`. Both are cheap and both have published
numbers.

---

## 5. Acquisition and licensing

- **Vendorable now (permissive licences, data already in the repos):** EnbPI's `Data/` (MIT),
  ELEC2 (present in three separate repos), Conformal PID's `tests/datasets/` (MIT), CopulaCPTS's
  `data/`. Suggest `scripts/fetch_ts_data.py` that shallow-clones each source repo, copies the
  relevant files into `data/timeseries/<group>/`, and writes a `meta.yaml` per series (frequency,
  seasonal period, target column, transform, upstream URL, licence). Record upstream commit SHAs —
  several of these repos have re-uploaded data files.
- **Needs an account/key:** NSRDB beyond the vendored extracts; LamaH-CE streamflow.
- **Needs credentialing (excluded):** MIMIC-III.
- **Link rot risk:** the Conformal PID COVID `deaths.csv` lives on a personal Google Drive. The
  per-state processed pickles are in the repo, so vendor those and note the rebuild path.
- **Attribution:** every group's `meta.yaml` should cite the paper that introduced it as a conformal
  benchmark *and* the original data provider — these are usually different (e.g. EnbPI popularised
  a series that UCI/NSRDB actually publish).

---

## 6. Config sketch

Slots directly into the group structure of `moc/configs/datasets.py`:

```python
# moc/configs/ts_datasets.py
xu_datasets = [
    'solar_atl', 'wind_hackberry', 'appliances', 'greenhouse', 'beijing_tiantan',
    'solar_palo_alto', 'solar_fremont', 'solar_milpitas', 'solar_mountain_view',
    'solar_north_san_jose', 'solar_redwood_city', 'solar_san_mateo',
    'solar_santa_clara', 'solar_sunnyvale',
]
angelopoulos_datasets = [
    'daily_climate', 'amzn', 'googl', 'msft',
    'covid_deaths_ak', 'covid_deaths_ca', 'covid_deaths_fl', 'covid_deaths_ga',
    'covid_deaths_ks', 'covid_deaths_ny', 'covid_deaths_tx',
]
# ... zaffran_datasets, auer_datasets, bhatnagar_datasets, elec2_datasets,
#     gibbs_datasets, sun_datasets, ts_synthetic_datasets

ts_real_groups = {
    'elec2': elec2_datasets,
    'xu': xu_datasets,
    'angelopoulos': angelopoulos_datasets,
    'auer': auer_datasets,
    'bhatnagar': bhatnagar_datasets,
    'gibbs': gibbs_datasets,
}
```

Per-series metadata that the datamodule needs and that `meta.yaml` must carry:
`freq`, `seasonal_period` (48 for ELEC2, 24 for hourly solar/air, 5 for daily stocks, 7 for daily
climate), `target_col`, `exog_cols`, `log_transform` (true for stock prices, per the PID configs),
`T_burnin`.

Adopt the literature's conventions where they are near-universal, so numbers are comparable
without rescaling: **α = 0.1** (not the repo's current 0.2), a burn-in period excluded from all
metrics, and horizon 1 by default with 24 reserved for the M4/NN5 group.

---

## 7. Positioning against existing reviews and benchmarks

Two recent works overlap with the intended contribution and should be read before scoping:

- **"A Gentle Introduction to Conformal Time Series Forecasting"** (arXiv 2511.13608, Nov 2025) — a
  review organised around reweighting calibration data / updating residual distributions /
  adaptively tuning target coverage, with a simulation study on coverage, width and cost. Its
  taxonomy is close to F2/F5/F3 above. **[web]**
- **"Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking"**
  (arXiv 2601.18509, Jan 2026) — an explicit benchmark, but scoped narrowly: AutoARIMA as the sole
  base forecaster, neural predictors deliberately excluded, evaluated on a monthly sales corpus
  with coverage / width / Winkler score. Reports that multi-step split CP is the most efficient
  method reaching 90%. **[web]**

Both were unreachable from this sandbox, so read them directly before finalising scope.

The differentiators available to this testbed, given what those two do not do:

1. **Score × scheme as a full cross-product.** Both existing works fix a score and vary the scheme.
   The whole point of inheriting this repository's design is that the score is a free axis, so you
   can answer "how much of the improvement attributed to method X is actually the score it happens
   to use?" — a question the literature currently cannot answer.
2. **Base-forecaster breadth.** The Jan-2026 benchmark explicitly excludes neural predictors; here
   naive → ARIMA → LGBM → quantile-LSTM → MDN → MQF2 is a declared axis.
3. **Non-convex prediction sets.** Density-based scores (`C-HDR`, `DR-CP`, `PCP`) yield unions of
   intervals in 1-D. No time-series benchmark evaluates these, and the exact grid-based measure in
   §4.1 of the plan makes them measurable.
4. **Miscoverage-dependence diagnostics.** Run-length and independence tests on the miscoverage
   sequence — the property split CP loses first and that almost no paper reports.
5. **Oracle conditional coverage** on the synthetic group.
6. **Reproduction of published results as a validation layer** (§4), rather than a fresh set of
   numbers with no tie-in to the literature.

---

## Sources

Verified by direct repository inspection:
[aangelopoulos/conformal-time-series](https://github.com/aangelopoulos/conformal-time-series) ·
[salesforce/online_conformal](https://github.com/salesforce/online_conformal) ·
[hamrel-cxu/EnbPI](https://github.com/hamrel-cxu/EnbPI) ·
[hamrel-cxu/SPCI-code](https://github.com/hamrel-cxu/SPCI-code) ·
[ml-jku/HopCPT](https://github.com/ml-jku/HopCPT) ·
[mzaffran/AdaptiveConformalPredictionsTimeSeries](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries) ·
[isgibbs/AdaptiveConformal](https://github.com/isgibbs/AdaptiveConformal) ·
[kamilest/conformal-rnn](https://github.com/kamilest/conformal-rnn) ·
[Rose-STL-Lab/CopulaCPTS](https://github.com/Rose-STL-Lab/CopulaCPTS)

From search:
[Conformal PID (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/47f2fad8c1111d07f83c91be7870f8db-Paper-Conference.pdf) ·
[ACI (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) ·
[Conformal prediction beyond exchangeability (AoS 2023)](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full) ·
[FACI/DtACI (JMLR 2024)](http://www.jmlr.org/papers/v25/22-1218.html) ·
[SAOCP (arXiv 2302.07869)](https://arxiv.org/abs/2302.07869) ·
[Split conformal prediction and non-exchangeable data (JMLR 2024)](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf) ·
[TQA (arXiv 2205.09940)](https://arxiv.org/abs/2205.09940) ·
[CPTD (arXiv 2205.12940)](https://arxiv.org/abs/2205.12940) ·
[Bellman Conformal Inference (arXiv 2402.05203)](https://arxiv.org/pdf/2402.05203) ·
[Online CP with decaying step sizes (arXiv 2402.01139)](https://arxiv.org/pdf/2402.01139) ·
[CPTC (arXiv 2509.02844)](https://openreview.net/forum?id=HgLaVgCpCl) ·
[A Gentle Introduction to Conformal Time Series Forecasting (arXiv 2511.13608)](https://arxiv.org/abs/2511.13608) ·
[CP Algorithms for TS Forecasting: Methods and Benchmarking (arXiv 2601.18509)](https://arxiv.org/abs/2601.18509)
