from .ensemble import BootstrapEnsemble
from .models import ts_models as _base_models
from .predictive import (
    DistributionPredictive,
    LocationScalePredictive,
    MixturePredictive,
    PointPredictive,
    QuantilePredictive,
)

ts_models = dict(_base_models)
ts_models['EnbPIEnsemble'] = BootstrapEnsemble

__all__ = [
    'ts_models',
    'BootstrapEnsemble',
    'PointPredictive',
    'QuantilePredictive',
    'DistributionPredictive',
    'LocationScalePredictive',
    'MixturePredictive',
]
