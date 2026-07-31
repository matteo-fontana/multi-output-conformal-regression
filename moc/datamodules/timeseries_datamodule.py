"""
Windowing datamodule for univariate time-series conformal prediction.

Deliberately *not* a subclass of `BaseDataModule`: that class's `load_datasets` is built around
`random_split`, which is precisely the operation that must not happen here, and its DataLoader
plumbing buys nothing for a pipeline whose conformal core is vectorised numpy. What is reused from
the existing architecture is everything above the datamodule -- `RunConfig`, the runner, the
hyperparameter DSL and the analysis stack.

The contract:

    train | gap | val | gap | calib | gap | test

all contiguous and in chronological order, with `gap = lags + horizon - 1` so that no window in a
later block overlaps a target in an earlier one.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from moc.configs.ts_datasets import get_spec
from moc.datamodules.ts_synthetic import generate

log = logging.getLogger('moc')


class Split:
    """A contiguous block of windowed observations."""

    def __init__(self, x, y, t_index, oracle_loc=None, oracle_scale=None):
        self.x = x
        self.y = y
        self.t_index = t_index
        self.oracle_loc = oracle_loc
        self.oracle_scale = oracle_scale

    def __len__(self):
        return len(self.y)

    @property
    def has_oracle(self):
        return self.oracle_loc is not None


def _calendar_features(t_index, period):
    if period is None or period <= 1:
        return np.zeros((len(t_index), 0)), []
    angle = 2 * np.pi * (t_index % period) / period
    return np.stack([np.sin(angle), np.cos(angle)], axis=1), ['cal_sin', 'cal_cos']


def make_windows(y, exog, lags, horizon, seasonal_period, exog_names=None):
    """Turn a raw series into a supervised design matrix.

    `x_t` holds `lags` autoregressive lags, the seasonal lag when it falls outside the lag window,
    contemporaneous exogenous variables, and calendar harmonics.
    Returns `(x, y_target, t_index, feature_names)` where `t_index` is the position of the target
    in the original series.
    """
    if horizon != 1:
        raise NotImplementedError(
            'Only one-step-ahead (horizon=1) is implemented. Multi-step joint bands are Phase 6 '
            'and reuse the existing multi-output conformalizers; see docs/TIMESERIES_TESTBED_PLAN.md.'
        )
    T = len(y)
    seasonal_lag = seasonal_period if (seasonal_period and seasonal_period > lags) else None
    start = max(lags, seasonal_lag or 0)
    t_index = np.arange(start, T - horizon + 1)
    if len(t_index) == 0:
        raise ValueError(f'Series of length {T} is too short for lags={lags}')

    cols, names = [], []
    for lag in range(1, lags + 1):
        cols.append(y[t_index - lag])
        names.append(f'lag_{lag}')
    if seasonal_lag is not None:
        cols.append(y[t_index - seasonal_lag])
        names.append(f'lag_seasonal_{seasonal_lag}')
    x = np.stack(cols, axis=1)

    if exog is not None and exog.shape[1] > 0:
        x = np.concatenate([x, exog[t_index]], axis=1)
        names += list(exog_names or [f'exog_{i}' for i in range(exog.shape[1])])

    cal, cal_names = _calendar_features(t_index, seasonal_period)
    if cal.shape[1] > 0:
        x = np.concatenate([x, cal], axis=1)
        names += cal_names

    return x, y[t_index], t_index, names


class TimeSeriesDataModule:
    """Loads a series, windows it, and splits it chronologically."""

    def __init__(self, rc):
        self.rc = rc
        self.config = rc.config
        self.spec = get_spec(rc.dataset_group, rc.dataset)
        self.lags = self.spec.lags or self.config.ts.lags
        self.horizon = self.config.ts.horizon
        self.seasonal_period = self.spec.seasonal_period
        self.load()

    # -- data acquisition ---------------------------------------------------------------------

    def _load_raw(self):
        """Returns `(y, exog, exog_names, series)` where `series` carries the oracle if any."""
        spec = self.spec
        if spec.group == 'ts_synthetic':
            T = 1500 if self.config.fast else self.config.ts.synthetic_length
            # run_id seeds the generator: the cleanest form of replication for synthetic data.
            series = generate(spec.name, T=T, seed=1000 * self.rc.run_id + 17)
            return series.y, series.exog, series.exog_names, series

        path = Path(self.config.data_dir) / spec.path
        if not path.exists():
            raise FileNotFoundError(
                f'{path} not found. Run `python scripts/fetch_ts_data.py --groups {spec.group}` '
                f'to vendor it from the upstream repository.'
            )
        df = pd.read_csv(path)
        if spec.skip_head:
            df = df.iloc[spec.skip_head:]
        target = spec.target if spec.target is not None else df.columns[-1]
        if target not in df.columns:
            raise ValueError(f'Target column {target!r} not in {path} (columns: {list(df.columns)})')

        exog_names = [c for c in spec.exog if c in df.columns]
        missing = set(spec.exog) - set(exog_names)
        if missing:
            log.warning(f'{spec.group}/{spec.name}: exogenous columns absent, skipping: {sorted(missing)}')

        keep = [target] + exog_names
        df = df[keep].apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill()
        df = df.dropna()

        y = df[target].to_numpy(dtype=np.float64)
        if spec.log_transform:
            if np.any(y <= 0):
                raise ValueError(f'{spec.name}: log_transform requested but series has non-positive values')
            y = np.log(y)
        exog = df[exog_names].to_numpy(dtype=np.float64) if exog_names else None
        return y, exog, exog_names, None

    # -- windowing and splitting --------------------------------------------------------------

    def load(self):
        y_raw, exog, exog_names, series = self._load_raw()

        # Truncate to the *most recent* max_size points. Contiguous, unlike BaseDataModule.subsample.
        max_size = 2000 if self.config.fast else self.config.ts.max_length
        if len(y_raw) > max_size:
            y_raw = y_raw[-max_size:]
            if exog is not None:
                exog = exog[-max_size:]
            if series is not None:
                series = _truncate_series(series, max_size)

        x, y, t_index, feature_names = make_windows(
            y_raw, exog, self.lags, self.horizon, self.seasonal_period, exog_names
        )
        self.feature_names = feature_names
        self.y_series = y_raw
        self.series = series

        oracle_loc = series.loc[t_index] if series is not None else None
        oracle_scale = series.scale[t_index] if series is not None else None
        self.oracle_dist = series.dist if series is not None else None
        self.oracle_df = series.df if series is not None else np.inf
        self.mixing_rate = series.mixing_rate if series is not None else None

        # Changepoints, expressed as positions within the windowed index.
        self.changepoints = None
        if series is not None and series.changepoints is not None:
            self.changepoints = np.searchsorted(t_index, series.changepoints)
            self.changepoints = self.changepoints[self.changepoints < len(t_index)]

        n = len(y)
        gap = self.lags + self.horizon - 1
        ratios = np.array(self.config.ts.split_ratio, dtype=float)
        ratios = ratios / ratios.sum()
        sizes = self._block_sizes(n, ratios, gap)
        self.split_sizes = sizes

        bounds, cursor = [], 0
        for i, size in enumerate(sizes):
            bounds.append((cursor, cursor + size))
            cursor += size + (gap if i < len(sizes) - 1 else 0)

        def make(lo, hi):
            return Split(
                x=x[lo:hi], y=y[lo:hi], t_index=t_index[lo:hi],
                oracle_loc=None if oracle_loc is None else oracle_loc[lo:hi],
                oracle_scale=None if oracle_scale is None else oracle_scale[lo:hi],
            )

        self.train, self.val, self.calib, self.test = [make(lo, hi) for lo, hi in bounds]
        self.test_offset = bounds[3][0]

        # Standardise features on the training block only.
        mu = self.train.x.mean(axis=0)
        sd = self.train.x.std(axis=0)
        sd[sd < 1e-8] = 1.0
        for split in (self.train, self.val, self.calib, self.test):
            split.x = (split.x - mu) / sd
        self.x_mean, self.x_std = mu, sd

        # Reported alongside widths so that sizes are comparable across datasets.
        self.y_scale = float(self.train.y.std()) or 1.0
        self.input_dim = x.shape[1]
        self.n_windows = n

        self._check_no_leakage(bounds, gap)
        log.info(
            f'{self.rc.dataset_group}/{self.rc.dataset}: T={len(y_raw)} windows={n} '
            f'split={[len(s) for s in (self.train, self.val, self.calib, self.test)]} gap={gap}'
        )

    def _block_sizes(self, n, ratios, gap):
        usable = n - 3 * gap
        min_needed = 4 * (self.config.ts.min_block_size)
        if usable < min_needed:
            raise ValueError(
                f'Series too short: {n} windows leaves {usable} usable after gaps, '
                f'need at least {min_needed}.'
            )
        sizes = np.maximum(
            (ratios * usable).astype(int), self.config.ts.min_block_size
        )
        # Cap the calibration block, keeping the most recent points (cf. BaseDataModule's 2048 cap).
        cap = self.config.ts.max_calib_size
        if sizes[2] > cap:
            sizes[0] += sizes[2] - cap
            sizes[2] = cap
        sizes[-1] = usable - sizes[:-1].sum()
        if sizes[-1] < self.config.ts.min_block_size:
            raise ValueError(f'Test block too small ({sizes[-1]}) for {self.rc.dataset}')
        return sizes

    def _check_no_leakage(self, bounds, gap):
        """Guard 1 of docs §8: strict chronological ordering with a wide enough gap."""
        for (_, prev_hi), (next_lo, _) in zip(bounds, bounds[1:]):
            assert next_lo - prev_hi >= gap, f'gap {next_lo - prev_hi} < required {gap}'
        for split, name in [(self.train, 'train'), (self.val, 'val'),
                            (self.calib, 'calib'), (self.test, 'test')]:
            assert len(split) > 0, f'{name} split is empty'
        assert self.train.t_index[-1] < self.val.t_index[0]
        assert self.val.t_index[-1] < self.calib.t_index[0]
        assert self.calib.t_index[-1] < self.test.t_index[0]

    # -- convenience --------------------------------------------------------------------------

    @property
    def has_oracle(self):
        return self.test.has_oracle

    def fit_data(self):
        """Training data for the base predictor: the train block, or train+val for models that
        do their own internal validation."""
        return self.train.x, self.train.y


def _truncate_series(series, max_size):
    from moc.datamodules.ts_synthetic import SyntheticSeries

    cps = None
    if series.changepoints is not None:
        shift = len(series.y) - max_size
        cps = series.changepoints - shift
        cps = cps[cps >= 0]
    return SyntheticSeries(
        y=series.y[-max_size:], loc=series.loc[-max_size:], scale=series.scale[-max_size:],
        dist=series.dist, df=series.df,
        exog=None if series.exog is None else series.exog[-max_size:],
        exog_names=series.exog_names, changepoints=cps, mixing_rate=series.mixing_rate,
    )
