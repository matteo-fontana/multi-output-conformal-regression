"""Registry of online calibration schemes.

Grouped by what they adapt, following the taxonomy of Stocker, Małgorzewicz, Fontana & Ben Taieb
(2025) -- see docs/TIMESERIES_METHODS_AND_DATASETS.md §1.
"""

from .aci import ACI, DtACI
from .base import (
    OnlineConformalizer,
    conformal_quantile,
    pinball_loss,
    pinball_loss_grad,
    weighted_conformal_quantile,
)
from .ogd import SAOCP, SFOGD
from .pid import PID, PIDScorecaster, QuantileTracker
from .residual_model import SPCI
from .split import NexCP, Rolling, Split

schemes = {
    'Split': Split,
    'Rolling': Rolling,
    'NexCP': NexCP,
    'ACI': ACI,
    'DtACI': DtACI,
    'SF-OGD': SFOGD,
    'SAOCP': SAOCP,
    'QuantileTracker': QuantileTracker,
    'PID': PID,
    'PID+Scorecaster': PIDScorecaster,
    'SPCI': SPCI,
}

# What each scheme adapts, for grouping in the results tables.
SCHEME_FAMILY = {
    'Split': 'static',
    'Rolling': 'calibration set',
    'NexCP': 'calibration weights',
    'ACI': 'target level',
    'DtACI': 'target level',
    'SF-OGD': 'threshold',
    'SAOCP': 'threshold',
    'QuantileTracker': 'threshold',
    'PID': 'threshold',
    'PID+Scorecaster': 'threshold',
    'SPCI': 'residual law',
}

__all__ = [
    'schemes', 'SCHEME_FAMILY', 'OnlineConformalizer',
    'conformal_quantile', 'weighted_conformal_quantile', 'pinball_loss', 'pinball_loss_grad',
    'Split', 'Rolling', 'NexCP', 'ACI', 'DtACI', 'SFOGD', 'SAOCP',
    'QuantileTracker', 'PID', 'PIDScorecaster', 'SPCI',
]
