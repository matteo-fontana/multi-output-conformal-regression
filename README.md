This is the repository associated with the paper [A Unified Comparative Study with Generalized Conformity Scores for Multi-Output Conformal Regression](https://arxiv.org/abs/2501.10533) (ICML 2025).

It includes:
- An implementation of several conformal methods for multi-output conformal regression.
- Several base predictors (Multivariate Quantile Function Forecaster, Distributional Random Forest, Gaussian Mixture parametrized by a hypernetwork).
- Metrics for marginal coverage, region size, and conditional coverage.
- A large empirical study based on datasets gathered from the literature, all with multiple outputs.

<p align="center">
<img src="images/MQF2_one_moon_heteroscedastic.png?raw=true" alt="" width="95%" align="top">
<img src="images/taxi_example.png?raw=true" alt="" width="95%" align="top">
</p>

## Datasets

All datasets except MEPS are directly available in this repository. See step 5 of the installation for downloading MEPS.

Refer to these repositories for more information on the datasets used in this study:
- https://github.com/tsoumakas/mulan
- https://github.com/Shai128/mqr
- https://github.com/lorismichel/drf
- https://github.com/Zhendong-Wang/Probabilistic-Conformal-Prediction
- https://github.com/aschnuecker/Superlevel-sets

## Example usage

The following code shows an example usage of the code in this repository.

```python
from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import L_CP


config = get_config()
config.device = 'cpu'
rc = RunConfig(config, 'mulan', 'sf2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim
model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)

alpha = 0.1
conformalizer = L_CP(datamodule.calib_dataloader(), model)
test_batch = next(iter(datamodule.test_dataloader()))
x, y = test_batch
coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
print(coverage)
print(volume)
```

## Installation

### Prerequisites
- Python (tested on 3.10.14)

<!--
echo 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' | sudo tee -a /etc/apt/sources.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo apt update
apt policy r-base
sudo apt install r-base
-->

### Steps
1. Clone the repository:
```bash
git clone https://github.com/Vekteur/multi-output-conformal-regression.git
cd multi-output-conformal-regression
```

2. (Optional) Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```
for exact versions ensuring reproducibility, or
```
pip install -r requirements.in
```
for more flexibility.

4. (Optional) If you want to run Distributional Random Forests, install R (tested on 4.4.1, version 4.1 or higher is required).
Open the R interpreter using the command `R` and run the following command:
```R
install.packages("drf")
```
Compilation will take a few minutes.
Then run
```bash
pip install --index-url https://test.pypi.org/simple/ drf==0.1
```

5. (Optional) For running experiments on the MEPS dataset, download it according to [these instructions](https://github.com/yromano/cqr/tree/master/get_meps_data), summarized below:
```bash
git clone https://github.com/yromano/cqr
cd cqr/get_meps_data/
Rscript download_data.R
python main_clean_and_save_to_csv.py
cd ../../
for id in 19 20 21; do mv "cqr/get_meps_data/meps_${id}_reg.csv" "data/feldman/meps_${id}.csv"; done
rm -rf cqr
```

## Reproducing the results

To generate the figures for toy datasets, run `toy_experiments.ipynb`.

To compute the results of the paper:
```
python run.py name="full" device="cuda" repeat_tuning=10
```
or use `device="cpu"` if you don't have a GPU.

To generate the figures based on these results, run `analysis.ipynb` 

## Time-series testbed

The repository also hosts a testbed for **univariate time-series conformal prediction**, built on
the same runner, hyperparameter DSL and analysis stack. Design notes are in
[`docs/TIMESERIES_TESTBED_PLAN.md`](docs/TIMESERIES_TESTBED_PLAN.md); the method survey and the
provenance of every dataset are in
[`docs/TIMESERIES_METHODS_AND_DATASETS.md`](docs/TIMESERIES_METHODS_AND_DATASETS.md).

The organising idea is that a conformal time-series method factorises into a **conformity score**
and a **calibration scheme**, and the two are varied independently:

| | |
|---|---|
| Scores | `abs_residual`, `signed_residual`, `normalized_residual`, `cqr`, `cqr_r`, `nll`, `hpd`, `pit` |
| Schemes | `Split`, `Rolling`, `NexCP`, `ACI`, `DtACI`, `SF-OGD`, `SAOCP`, `QuantileTracker`, `PID`, `PID+Scorecaster`, `SPCI` |
| Base models | `SeasonalNaive`, `Ridge`, `LGBM`, `LGBMQuantile`, `LinearQuantile`, `GaussianRidge`, `RFKDE`, `EnbPIEnsemble`, `Oracle` |

Pairing `EnbPIEnsemble` with `Rolling` reproduces EnbPI; the other pairings are combinations the
literature has not reported.

### Getting the data

```bash
python scripts/fetch_ts_data.py
```

This vendors 25 real series from the original authors' MIT-licensed repositories (EnbPI, Conformal
PID, AgACI), writing `data/timeseries/<group>/<name>.csv` plus a `meta.yaml` recording provenance
and the upstream commit. Eleven synthetic generators with exact conditional laws need no download.

### Running

```bash
# smoke test
python run_ts.py name="ts_test" datasets="ts_test" tuning_type="timeseries_test" repeat_tuning=1

# the default working set: 9 datasets covering drift, changepoints, volatility and seasonality
python run_ts.py name="ts_full" datasets="ts_filtered" repeat_tuning=10 nb_workers=8

# everything, with the full hyperparameter grid
python run_ts.py name="ts_all" datasets="ts_all" tuning_type="timeseries_full" repeat_tuning=10
```

`alpha` defaults to 0.1 here (the convention in this literature) rather than the 0.2 used by the
multi-output study. Replication uses `run_id` as a rolling-origin fold rather than a random
re-split, since a time series cannot be randomly re-split.

`SPCI` and `PID+Scorecaster` train models inside the online loop; enable them with
`ts.allow_sequential_model_methods=true`.

### Analysis

```python
from moc.analysis.dataframes import load_config
from moc.analysis.ts_dataframes import load_ts_results, summary_table, rank_table
from moc.analysis.plot_cd_diagram import draw_my_cd_diagram

df = load_ts_results(load_config('logs/ts_full'))
print(summary_table(df, metrics=('coverage', 'lce_100', 'width', 'winkler')))
print(rank_table(df, 'lce_100'))
```

Metrics are grouped as validity / efficiency / compute. Beyond marginal coverage and width, the
testbed reports local (rolling-window) coverage error, miscoverage run-length and a runs test,
exact set measure with connected-component counts for the non-convex sets that density-based
scores produce, Winkler and pinball scores, adaptivity correlations, regret against the best fixed
threshold, and — on the synthetic group — exact conditional coverage against the known law.

### Tests

```bash
python -m pytest tests/test_ts_conformal.py
```

The suite is built around the failure modes that produce plausible-but-wrong numbers rather than
crashes: chronological split integrity, causality of every online update (perturbing the tail of
the score stream must not change earlier thresholds), agreement between the grid-based and
score-based coverage computations, and the weighted-quantile normalisation.
