"""
Registry of time-series datasets for the conformal time-series testbed.

Groups are named after the first author of the paper that introduced the series as a conformal
benchmark, mirroring the convention already used in `moc/configs/datasets.py` (camehl, cevid,
del_barrio, feldman, mulan, wang).

See docs/TIMESERIES_METHODS_AND_DATASETS.md for provenance of every entry.
"""

from dataclasses import dataclass, field


@dataclass
class TSDatasetSpec:
    """Everything the datamodule needs to turn a raw file into a supervised windowed problem."""

    group: str
    name: str
    # Path relative to `config.data_dir`. None means the series is generated, not loaded.
    path: str = None
    # Column holding the target. None means "single unnamed column".
    target: str = None
    # Exogenous columns used as contemporaneous features.
    exog: list = field(default_factory=list)
    # Seasonal period in time steps, used for seasonal-naive and calendar features.
    seasonal_period: int = 1
    # Some series (prices) are modelled on the log scale, as in the Conformal PID configs.
    log_transform: bool = False
    # Optional per-dataset override of the number of autoregressive lags.
    lags: int = None
    # Rows to drop from the head of the file (e.g. leading NaNs after differencing upstream).
    skip_head: int = 0
    # Human-readable provenance, surfaced in the analysis tables.
    source: str = ''
    # Whether the series appears in the group listings. Vendored-but-excluded series stay
    # addressable by name while being kept out of default sweeps (see the COVID entries).
    in_groups: bool = True


# --------------------------------------------------------------------------------------------
# Synthetic (oracle conditional law available; see moc/datamodules/ts_synthetic.py)
# --------------------------------------------------------------------------------------------

SYNTHETIC_NAMES = [
    'ar1_gauss',           # sanity check: exchangeable-ish, every method should tie
    'ar1_garch',           # separates volatility-adaptive scores from plain absolute residuals
    'ar1_heavytail',       # Student-t innovations, stresses width metrics
    'changepoint_mean',    # abrupt level shift
    'changepoint_var',     # abrupt variance shift, the classic ACI motivation
    'regime_switch',       # Markov-switching mean and variance
    'slow_drift',          # time-varying AR coefficient
    'seasonal_hetero',     # seasonality + trend + heteroscedasticity
    'adversarial',         # worst case for exchangeability
    'zaffran_ar09',        # AgACI's AR(1) with phi = 0.9 (their sign convention)
    'zaffran_arma',        # AgACI's ARMA(1,1) setting
]


# --------------------------------------------------------------------------------------------
# Real series
# --------------------------------------------------------------------------------------------

# ELEC2 -- the field's anchor dataset. Vendored identically in the SPCI, Conformal PID and
# NexCP-adjacent repositories. Half-hourly, New South Wales, 1996-05-07 to 1998-12-05.
ELEC2_SPECS = [
    TSDatasetSpec(
        group='elec2', name='elec2', path='timeseries/elec2/elec2.csv',
        target='nswdemand', exog=['nswprice', 'vicprice', 'vicdemand', 'transfer'],
        seasonal_period=48, source='Harries 1999; via aangelopoulos/conformal-time-series',
    ),
    # DtACI's framing: predict `transfer` from the two states' prices and demands.
    TSDatasetSpec(
        group='elec2', name='elec2_transfer', path='timeseries/elec2/elec2.csv',
        target='transfer', exog=['nswprice', 'nswdemand', 'vicprice', 'vicdemand'],
        seasonal_period=48, source='Gibbs & Candes 2024 framing',
    ),
]

# Xu & Xie's EnbPI / SPCI benchmark suite.
_XU_SOLAR_SITES = [
    'palo_alto', 'fremont', 'milpitas', 'mountain_view', 'north_san_jose',
    'redwood_city', 'san_mateo', 'santa_clara', 'sunnyvale',
]

XU_SPECS = [
    TSDatasetSpec(
        group='xu', name='solar_atl', path='timeseries/xu/solar_atl.csv',
        target='DHI', exog=['DNI', 'Dew Point', 'Temperature', 'Wind Speed', 'Relative Humidity',
                            'Solar Zenith Angle'],
        seasonal_period=24, source='NSRDB Atlanta hourly; EnbPI (Xu & Xie 2021)',
    ),
    TSDatasetSpec(
        group='xu', name='wind_hackberry', path='timeseries/xu/wind_hackberry.csv',
        target='MWH', seasonal_period=24,
        source='Hackberry wind generation 2019-2020; EnbPI',
    ),
    TSDatasetSpec(
        group='xu', name='appliances', path='timeseries/xu/appliances.csv',
        target='Appliances', exog=['lights', 'T_out', 'RH_out', 'Windspeed', 'Tdewpoint'],
        seasonal_period=144, source='UCI Appliances Energy Prediction (10-min); EnbPI',
    ),
    TSDatasetSpec(
        group='xu', name='beijing_tiantan', path='timeseries/xu/beijing_tiantan.csv',
        target='PM2.5', exog=['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM'],
        seasonal_period=24, source='UCI Beijing Multi-Site Air Quality, Tiantan; EnbPI/HopCPT',
    ),
] + [
    TSDatasetSpec(
        group='xu', name=f'solar_{site}', path=f'timeseries/xu/solar_{site}.csv',
        target='DHI', exog=['DNI', 'Dew Point', 'Temperature', 'Wind Speed',
                            'Relative Humidity', 'Solar Zenith Angle'],
        seasonal_period=24, source='NSRDB Bay Area 2018 hourly; EnbPI (TPAMI version)',
    )
    for site in _XU_SOLAR_SITES
]

# Angelopoulos, Candes & Tibshirani's Conformal PID benchmark suite.
_ANGELOPOULOS_STOCKS = ['amzn', 'googl', 'msft']
_ANGELOPOULOS_STATES = ['ak', 'ca', 'fl', 'ga', 'ks', 'ny', 'tx']

ANGELOPOULOS_SPECS = [
    TSDatasetSpec(
        group='angelopoulos', name='daily_climate', path='timeseries/angelopoulos/daily_climate.csv',
        target='meantemp', exog=['humidity', 'wind_speed', 'meanpressure'],
        seasonal_period=7, source='Delhi daily climate; Conformal PID',
    ),
] + [
    # `log: True` in the upstream configs; seasonal period 5 for trading days.
    TSDatasetSpec(
        group='angelopoulos', name=ticker, path=f'timeseries/angelopoulos/{ticker}.csv',
        target='Open', seasonal_period=5, log_transform=True,
        source='DJIA panel daily open from 2006-01-03; Conformal PID',
    )
    for ticker in _ANGELOPOULOS_STOCKS
] + [
    # Vendored and addressable, but kept out of the group listings: 135 weekly points is too short
    # for a four-way chronological split to give a meaningful test block, and the task in the
    # source paper is 4-week-ahead anyway, so it belongs to the horizon > 1 setting (Phase 6).
    TSDatasetSpec(
        group='angelopoulos', name=f'covid_deaths_{state}',
        path=f'timeseries/angelopoulos/covid_deaths_{state}.csv',
        target='y', seasonal_period=1, lags=4, in_groups=False,
        source='4-week-ahead statewide COVID death forecasting; Conformal PID',
    )
    for state in _ANGELOPOULOS_STATES
]

# Zaffran et al.'s French electricity spot prices (eco2mix), 2016-2019 hourly.
ZAFFRAN_SPECS = [
    TSDatasetSpec(
        group='zaffran', name='spot_france', path='timeseries/zaffran/spot_france.csv',
        target='Spot', seasonal_period=24,
        source='French spot prices 2016-2019 via eco2mix; AgACI (Zaffran et al. 2022)',
    ),
]

REAL_SPECS = ELEC2_SPECS + XU_SPECS + ANGELOPOULOS_SPECS + ZAFFRAN_SPECS

SYNTHETIC_SPECS = [
    TSDatasetSpec(group='ts_synthetic', name=name, seasonal_period=24 if 'seasonal' in name else 1,
                  source='generated; see moc/datamodules/ts_synthetic.py')
    for name in SYNTHETIC_NAMES
]

ALL_SPECS = SYNTHETIC_SPECS + REAL_SPECS
SPEC_BY_KEY = {(spec.group, spec.name): spec for spec in ALL_SPECS}


def get_spec(group, name):
    key = (group, name)
    if key not in SPEC_BY_KEY:
        raise ValueError(f'Unknown time-series dataset {group}/{name}')
    return SPEC_BY_KEY[key]


def _names(specs, group):
    return [spec.name for spec in specs if spec.group == group and spec.in_groups]


ts_synthetic_groups = {'ts_synthetic': SYNTHETIC_NAMES}

ts_real_groups = {
    group: _names(REAL_SPECS, group)
    for group in ['elec2', 'xu', 'angelopoulos', 'zaffran']
}

ts_all_groups = ts_synthetic_groups | ts_real_groups

# The default working set: covers every mechanism (volatility, changepoint, drift, seasonality,
# heavy tails, real non-stationarity) at a cost that fits a laptop-scale sweep.
ts_filtered_groups = {
    'ts_synthetic': ['ar1_garch', 'changepoint_var', 'slow_drift'],
    'elec2': ['elec2'],
    'xu': ['solar_atl', 'wind_hackberry', 'beijing_tiantan'],
    'angelopoulos': ['daily_climate', 'amzn'],
}

ts_test_groups = {'ts_synthetic': ['ar1_garch']}


def get_ts_dataset_groups(key):
    if key in ('ts_default', 'ts_filtered'):
        return ts_filtered_groups
    if key == 'ts_all':
        return ts_all_groups
    if key == 'ts_synthetic':
        return ts_synthetic_groups
    if key == 'ts_real':
        return ts_real_groups
    if key == 'ts_test':
        return ts_test_groups
    if key in ts_all_groups:
        return {key: ts_all_groups[key]}
    raise ValueError(f'Unknown time-series dataset group {key}')


def is_ts_key(key):
    return key.startswith('ts_') or key in ts_all_groups


def is_ts_group(group):
    return group in ts_all_groups
