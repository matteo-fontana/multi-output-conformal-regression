"""
Entry point for the time-series conformal testbed. Mirrors `run.py`.

    python run_ts.py name="ts_test" datasets="ts_test" tuning_type="timeseries_test" repeat_tuning=1
    python run_ts.py name="ts_full" datasets="ts_filtered" repeat_tuning=10 nb_workers=8

Defaults that differ from the multi-output pipeline and are set here rather than in
`general_config`, so the ICML entry point keeps its own conventions:

    alpha        0.1  (the convention across the conformal time-series literature)
    datasets     ts_filtered
    tuning_type  timeseries
"""

import sys
import warnings
from pathlib import Path

from dask.distributed import Client
from omegaconf import OmegaConf

from moc import utils
from moc.configs.config import get_config
from moc.runner import run_all


TS_DEFAULTS = {
    'alpha': 0.1,
    'datasets': 'ts_filtered',
    'tuning_type': 'timeseries',
    'manager': 'joblib',
}


def main():
    cli = OmegaConf.from_cli(sys.argv)
    config = OmegaConf.merge(OmegaConf.create(TS_DEFAULTS), cli)
    config = get_config(config)
    OmegaConf.resolve(config)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    if config.get('print_config'):
        utils.print_config(config, resolve=True)

    manager = config.manager
    if config.nb_workers == 1:
        manager = 'sequential'
    elif manager == 'dask':
        Client(n_workers=config.nb_workers, threads_per_worker=1, memory_limit=None)

    return run_all(config, manager=manager)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()
