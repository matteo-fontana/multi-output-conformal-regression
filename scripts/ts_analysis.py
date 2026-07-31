#!/usr/bin/env python
"""
Turn a time-series sweep into the tables and figures a review paper needs.

    python scripts/ts_analysis.py logs/ts_full
    python scripts/ts_analysis.py logs/ts_full --by score --out report/

Emits, under the validity / efficiency / compute headings of Stocker et al. (2025):

- a per-method summary (mean and standard deviation over datasets and folds);
- mean-rank tables, the input a critical-difference diagram summarises;
- the validity-efficiency frontier, which is the trade-off the review is about;
- the score x scheme grid for a chosen metric -- the cell-by-cell view that
  existing benchmarks cannot produce because they hold the score fixed;
- CD diagrams as PNGs, via the existing `plot_cd_diagram` module.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from moc.analysis.dataframes import load_config  # noqa: E402
from moc.analysis.ts_dataframes import (  # noqa: E402
    load_ts_results, rank_table, summary_table, to_long, validity_efficiency_frontier,
)

HEADLINE = ['coverage', 'lce_100', 'width', 'winkler', 'longest_miss_run', 'total_time']
CD_METRICS = ['lce_100', 'width', 'winkler', 'coverage']


def section(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def score_scheme_grid(df, metric):
    """The cross-product view: rows are scores, columns are schemes."""
    return df.pivot_table(index='score', columns='scheme', values=metric, aggfunc='mean')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log_dir')
    ap.add_argument('--by', nargs='+', default=['score', 'scheme'],
                    help='columns that define a "method" on the diagrams')
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--out', default=None, help='directory for figures')
    args = ap.parse_args()

    config = load_config(args.log_dir)
    df = load_ts_results(config)
    by = tuple(args.by)

    section('coverage')
    print(f'{len(df)} rows | {df.dataset.nunique()} datasets | {df.model.nunique()} models '
          f'| {df.score.nunique()} scores | {df.scheme.nunique()} schemes '
          f'| {df.run_id.nunique()} folds')
    missing = df[HEADLINE].isna().mean()
    if missing.any():
        print('missing values per metric:')
        print(missing[missing > 0].to_string())

    section(f'per-method summary (mean over datasets x folds), grouped by {by}')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(summary_table(df, metrics=tuple(HEADLINE), by=by).round(4).to_string())

    section('mean ranks (lower is better; coverage folded to |cov - (1-alpha)|)')
    for metric in CD_METRICS:
        print(f'\n-- {metric}')
        print(rank_table(df, metric, alpha=args.alpha, by=by).round(2).to_string())

    section('validity-efficiency frontier (lce_100 vs width)')
    with pd.option_context('display.max_rows', 200):
        print(validity_efficiency_frontier(df, by=by).round(4).to_string())

    section('score x scheme grid')
    for metric in ['lce_100', 'width', 'coverage']:
        print(f'\n-- {metric}')
        print(score_scheme_grid(df, metric).round(4).to_string())

    section('by scheme family')
    print(df.groupby('scheme_family')[HEADLINE].mean().round(4).to_string())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        from moc.analysis.plot_cd_diagram import draw_my_cd_diagram
        long = to_long(df, metrics=CD_METRICS, by=by)
        long['name'] = long['name'].astype(str)
        for metric in CD_METRICS:
            try:
                draw_my_cd_diagram(long, metric, args.alpha)
                path = out / f'cd_{metric}.png'
                plt.savefig(path, bbox_inches='tight', dpi=150)
                plt.close('all')
                print(f'wrote {path}')
            except Exception as e:
                print(f'CD diagram for {metric} failed: {type(e).__name__}: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
