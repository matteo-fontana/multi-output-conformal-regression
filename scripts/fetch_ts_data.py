#!/usr/bin/env python
"""
Vendor the time-series benchmark datasets from the original authors' repositories.

Every series here is one that a published conformal time-series method benchmarks on, so that the
testbed can reproduce those results as a validation layer before generating novel ones
(docs/TIMESERIES_METHODS_AND_DATASETS.md §4). All the sources below are MIT-licensed repositories
that vendor their own data.

    python scripts/fetch_ts_data.py                      # everything obtainable
    python scripts/fetch_ts_data.py --groups elec2 xu    # a subset
    python scripts/fetch_ts_data.py --keep-clones        # leave the clones for inspection

Each series is written as `data/timeseries/<group>/<name>.csv` with a sibling `meta.yaml`
recording provenance, the upstream commit, and the licence. Commit SHAs matter: several of these
repositories have re-uploaded their data files.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

REPOS = {
    'enbpi': ('https://github.com/hamrel-cxu/EnbPI.git', 'main', 'MIT'),
    'pid': ('https://github.com/aangelopoulos/conformal-time-series.git', 'main', 'MIT'),
    'agaci': ('https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries.git', 'main', 'MIT'),
}

BAY_AREA_SITES = {
    'palo_alto': 'Palo_Alto', 'fremont': 'Fremont', 'milpitas': 'Milpitas',
    'mountain_view': 'Mountain_View', 'north_san_jose': 'North_San_Jose',
    'redwood_city': 'Redwood_City', 'san_mateo': 'San_Mateo',
    'santa_clara': 'Santa_Clara', 'sunnyvale': 'Sunnyvale',
}

COVID_STATES = ['ak', 'ca', 'fl', 'ga', 'ks', 'ny', 'tx']
STOCKS = {'amzn': 'AMZN', 'googl': 'GOOGL', 'msft': 'MSFT'}


def clone(url, branch, dest):
    print(f'  cloning {url}')
    subprocess.run(
        ['git', 'clone', '--depth', '1', '--branch', branch, '-q', url, str(dest)],
        check=True,
    )
    sha = subprocess.run(['git', '-C', str(dest), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=True).stdout.strip()
    return sha


def write(df, out_dir, name, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f'{name}.csv', index=False)
    with open(out_dir / f'{name}.meta.yaml', 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f'    {name:24s} {df.shape[0]:>7d} rows  {list(df.columns)[:5]}')


# ------------------------------------------------------------------------------------------------
# Group builders
# ------------------------------------------------------------------------------------------------

def build_elec2(repos, data_dir, shas):
    src = repos['pid'] / 'tests' / 'datasets' / 'elec2.csv'
    df = pd.read_csv(src)
    meta = dict(
        source='Harries 1999, NSW electricity market; half-hourly 1996-05-07 to 1998-12-05',
        vendored_from='aangelopoulos/conformal-time-series', commit=shas['pid'], licence='MIT',
        used_by=['NexCP (Barber et al. 2023)', 'DtACI (Gibbs & Candes 2024)',
                 'SPCI (Xu & Xie 2023)', 'Conformal PID (Angelopoulos et al. 2023)'],
        freq='30min', seasonal_period=48,
    )
    write(df, data_dir / 'elec2', 'elec2', meta)


def build_xu(repos, data_dir, shas):
    base = repos['enbpi'] / 'Data'
    out = data_dir / 'xu'
    common = dict(vendored_from='hamrel-cxu/EnbPI', commit=shas['enbpi'], licence='MIT',
                  used_by=['EnbPI (Xu & Xie 2021)', 'SPCI (Xu & Xie 2023)'])

    # NSRDB exports carry two metadata rows before the real header.
    solar = pd.read_csv(base / 'Solar_Atl_data.csv', skiprows=2)
    solar = solar.loc[:, ~solar.columns.str.startswith('Unnamed')]
    write(solar, out, 'solar_atl', dict(
        source='NSRDB Atlanta (33.76, -84.39), hourly 2018', freq='1h', seasonal_period=24,
        **common))

    wind = pd.read_csv(base / 'Wind_Hackberry_Generation_2019_2020.csv')
    write(wind, out, 'wind_hackberry', dict(
        source='Hackberry wind farm generation, hourly 2019-2020', freq='1h', seasonal_period=24,
        **common))

    app = pd.read_csv(base / 'appliances_data.csv')
    write(app, out, 'appliances', dict(
        source='UCI Appliances Energy Prediction, 10-minute', freq='10min', seasonal_period=144,
        **common))

    air = pd.read_csv(base / 'Beijing_air_Tiantan_data.csv')
    write(air, out, 'beijing_tiantan', dict(
        source='UCI Beijing Multi-Site Air Quality, Tiantan station, hourly 2013-2017',
        freq='1h', seasonal_period=24,
        used_by=['EnbPI (Xu & Xie 2021)', 'HopCPT (Auer et al. 2023)'],
        vendored_from='hamrel-cxu/EnbPI', commit=shas['enbpi'], licence='MIT'))

    for name, fname in BAY_AREA_SITES.items():
        path = base / f'{fname}_data.csv'
        if not path.exists():
            print(f'    (skipping {name}: {path.name} absent)')
            continue
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        if df.columns[0] == '':
            df = df.iloc[:, 1:]
        write(df, out, f'solar_{name}', dict(
            source=f'NSRDB {fname.replace("_", " ")} 2018, hourly', freq='1h', seasonal_period=24,
            used_by=['EnbPI (TPAMI version)'],
            vendored_from='hamrel-cxu/EnbPI', commit=shas['enbpi'], licence='MIT'))


def build_angelopoulos(repos, data_dir, shas):
    base = repos['pid'] / 'tests' / 'datasets'
    out = data_dir / 'angelopoulos'
    common = dict(vendored_from='aangelopoulos/conformal-time-series', commit=shas['pid'],
                  licence='MIT', used_by=['Conformal PID (Angelopoulos et al. 2023)'])

    climate = pd.read_csv(base / 'daily-climate.csv')
    climate = climate.loc[:, ~climate.columns.str.startswith('Unnamed')]
    write(climate, out, 'daily_climate', dict(
        source='Delhi daily climate', freq='1D', seasonal_period=7, **common))

    djia = pd.read_csv(base / 'djia.csv')
    for name, ticker in STOCKS.items():
        sub = djia[djia['Name'] == ticker][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        sub = sub.sort_values('Date').reset_index(drop=True)
        if sub.empty:
            print(f'    (skipping {name}: ticker {ticker} absent from djia.csv)')
            continue
        write(sub, out, name, dict(
            source=f'{ticker} daily open from the DJIA panel, from 2006-01-03',
            freq='1B', seasonal_period=5, log_transform=True, **common))

    for state in COVID_STATES:
        path = base / 'covid-ts-proc' / 'statewide' / f'{state}_proc_4wkdeaths.pkl'
        if not path.exists():
            print(f'    (skipping covid_deaths_{state}: {path.name} absent)')
            continue
        df = pd.read_pickle(path)
        # The pickle is long-form; the observed series is the `y` variable.
        obs = df[df['variable'] == 'y'] if 'variable' in df.columns else df
        obs = obs.sort_values('timestamp')
        write(pd.DataFrame({'timestamp': obs['timestamp'].values, 'y': obs['target'].values}),
              out, f'covid_deaths_{state}', dict(
                  source=f'4-week-ahead COVID death forecasting, {state.upper()}',
                  freq='1W', seasonal_period=1, **common))


def build_zaffran(repos, data_dir, shas):
    src = repos['agaci'] / 'data_prices' / 'Prices_2016_2019_extract.csv'
    df = pd.read_csv(src)
    keep = [c for c in df.columns if c in ('Date', 'Spot', 'hour')]
    write(df[keep], data_dir / 'zaffran', 'spot_france', dict(
        source='French electricity spot prices 2016-2019, built from eco2mix',
        vendored_from='mzaffran/AdaptiveConformalPredictionsTimeSeries', commit=shas['agaci'],
        licence='MIT', used_by=['AgACI (Zaffran et al. 2022)'],
        freq='1h', seasonal_period=24))


BUILDERS = {
    'elec2': (build_elec2, ['pid']),
    'xu': (build_xu, ['enbpi']),
    'angelopoulos': (build_angelopoulos, ['pid']),
    'zaffran': (build_zaffran, ['agaci']),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groups', nargs='*', default=sorted(BUILDERS),
                    choices=sorted(BUILDERS), help='which dataset groups to build')
    ap.add_argument('--data-dir', default='data/timeseries')
    ap.add_argument('--keep-clones', action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    needed = sorted({r for g in args.groups for r in BUILDERS[g][1]})
    tmp = Path(tempfile.mkdtemp(prefix='ts_data_'))
    repos, shas = {}, {}
    try:
        for key in needed:
            url, branch, _ = REPOS[key]
            dest = tmp / key
            shas[key] = clone(url, branch, dest)
            repos[key] = dest
        for group in args.groups:
            print(f'building {group}')
            BUILDERS[group][0](repos, data_dir, shas)
    finally:
        if args.keep_clones:
            print(f'clones left in {tmp}')
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f'\ndone -> {data_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
