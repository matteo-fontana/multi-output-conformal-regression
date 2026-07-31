#!/usr/bin/env python
"""
Reproduction gate: match published results before trusting the testbed's novel cells.

Two cheap checks with numbers in the literature (docs/TIMESERIES_TESTBED_PLAN.md, Phase 4b):

1. **SPCI vs EnbPI on ELEC2.** Xu & Xie (2023) report that modelling the residual process shrinks
   the interval width substantially relative to EnbPI at matched coverage; their repository ships
   this exact comparison as `tutorial_electric_EnbPI_SPCI.ipynb`. Here EnbPI is the bootstrap
   ensemble paired with the sliding-window scheme, and SPCI is the same score with the
   quantile-forest scheme -- the separation the testbed is built around.

2. **ACI's learning-rate sensitivity on AgACI's AR(1).** Zaffran et al. (2022) motivate AgACI by
   showing that ACI's behaviour depends materially on gamma, which is not knowable in advance.
   Sweeping gamma should reproduce that spread, and DtACI -- which aggregates over gamma -- should
   land inside it without being told the right value.

    python scripts/reproduce_published.py
    python scripts/reproduce_published.py --check spci
"""

import argparse
import sys

import numpy as np

from moc.configs.config import get_config
from moc.datamodules.timeseries_datamodule import TimeSeriesDataModule
from moc.metrics.ts_evaluator import TSEvaluator
from moc.models.ts import ts_models
from moc.utils.run_config import RunConfig

ALPHA = 0.1


def build(dataset, group, run_id=0, **ts_overrides):
    config = get_config()
    config.alpha = ALPHA
    for k, v in ts_overrides.items():
        config.ts[k] = v
    rc = RunConfig(config=config, dataset_group=group, dataset=dataset, run_id=run_id)
    return TimeSeriesDataModule(rc)


def check_spci_vs_enbpi(grid_size=1000):
    print('\n=== 1. SPCI vs EnbPI on ELEC2 (Xu & Xie 2023) ===')
    dm = build('elec2', 'elec2')
    print(f'    windows={dm.n_windows} test={len(dm.test)} features={dm.input_dim}')

    rows = []
    ensemble = ts_models['EnbPIEnsemble'](base='Ridge', n_bootstrap=20)
    ev = TSEvaluator(dm, ensemble, ALPHA, grid_size=grid_size)
    for label, scheme, kwargs in [
        ('EnbPI (ensemble + rolling)', 'Rolling', {'window': 500}),
        ('SPCI  (ensemble + QRF)', 'SPCI', {'lag': 10, 'window': 500, 'stride': 25}),
        ('Split (control)', 'Split', {}),
    ]:
        r = ev.evaluate('abs_residual', scheme, scheme_kwargs=kwargs)
        rows.append((label, r))

    _print_table(rows, ['coverage', 'width', 'winkler', 'lce_100', 'update_time'])

    enbpi = dict(rows)['EnbPI (ensemble + rolling)']
    spci = dict(rows)['SPCI  (ensemble + QRF)']
    change = 100 * (spci['width'] - enbpi['width']) / enbpi['width']
    print(f'    SPCI width vs EnbPI: {change:+.1f}%   '
          f'(published direction: narrower at matched coverage)')
    ok = spci['coverage'] > 1 - ALPHA - 0.03 and change < 0
    print(f'    -> {"MATCHES" if ok else "DOES NOT MATCH"} the published direction')
    return ok


def check_aci_gamma_sensitivity(grid_size=800):
    print("\n=== 2. ACI learning-rate sensitivity on AgACI's AR(1), phi=0.9 (Zaffran et al. 2022) ===")
    dm = build('zaffran_ar09', 'ts_synthetic')
    ev = TSEvaluator(dm, ts_models['Ridge'](), ALPHA, grid_size=grid_size)

    rows = []
    for gamma in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        r = ev.evaluate('abs_residual', 'ACI', scheme_kwargs={'gamma': gamma})
        rows.append((f'ACI gamma={gamma:<7g}', r))
    rows.append(('DtACI (aggregates gamma)', ev.evaluate('abs_residual', 'DtACI')))
    rows.append(('Split (control)', ev.evaluate('abs_residual', 'Split')))

    _print_table(rows, ['coverage', 'width', 'lce_100', 'longest_miss_run'])

    widths = [r['width'] for label, r in rows if label.startswith('ACI')]
    spread = 100 * (max(widths) - min(widths)) / np.mean(widths)
    dtaci = dict(rows)['DtACI (aggregates gamma)']
    print(f'    width spread across gamma: {spread:.1f}% of the mean')
    print(f'    DtACI width {dtaci["width"]:.3f} sits in [{min(widths):.3f}, {max(widths):.3f}]: '
          f'{min(widths) <= dtaci["width"] <= max(widths)}')
    ok = spread > 1.0 and min(widths) <= dtaci['width'] <= max(widths)
    print(f'    -> {"MATCHES" if ok else "DOES NOT MATCH"} the published motivation for aggregating')
    return ok


def _print_table(rows, metrics):
    header = f'    {"method":<28s}' + ''.join(f'{m:>14s}' for m in metrics)
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for label, r in rows:
        cells = ''.join(f'{r.get(m, float("nan")):>14.4f}' for m in metrics)
        print(f'    {label:<28s}{cells}')


CHECKS = {'spci': check_spci_vs_enbpi, 'aci': check_aci_gamma_sensitivity}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', nargs='*', default=sorted(CHECKS), choices=sorted(CHECKS))
    args = ap.parse_args()

    results = {name: CHECKS[name]() for name in args.check}
    print('\n=== summary ===')
    for name, ok in results.items():
        print(f'    {name:<6s} {"PASS" if ok else "FAIL"}')
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
