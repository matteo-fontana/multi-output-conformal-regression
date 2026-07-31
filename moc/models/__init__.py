"""Model and trainer registries.

`models` resolves lazily: the multi-output predictors pull in heavy optional dependencies
(`cpflows` for MQF2, `rpy2`/`drf` for DRF+KDE), and importing this package must not require them
when only the time-series testbed is in use.

`trainers` stays eager. It cannot be lazy: `moc.models.trainers` is also a subpackage, so
importing a trainer module rebinds the name on this package and would silently replace the
registry with the subpackage. Importing them up front sets that attribute first and lets the
dict shadow it, which is what the original module did.
"""

from importlib import import_module

from moc.models.trainers.default_trainer import DefaultTrainer
from moc.models.trainers.lightning_trainer import get_lightning_trainer


class _LazyRegistry(dict):
    """A dict whose values are `(module, attribute)` pairs, imported on first access."""

    def __getitem__(self, key):
        module, attr = super().__getitem__(key)
        return getattr(import_module(module), attr)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


models = _LazyRegistry({
    'Oracle': ('moc.models.oracle.oracle_model', 'OracleModel'),
    'DRF-KDE': ('moc.models.drf_kde.drf_kde', 'DRF_KDE'),
    'MQF2': ('moc.models.mqf2.lightning_module', 'MQF2LightningModule'),
    'Mixture': ('moc.models.mixture.mixture_model', 'MixtureLightningModule'),
    'Glow': ('moc.models.glow.glow', 'GlowPreTrained'),
    'Quantile': ('moc.models.quantile.quantile_model', 'QuantileModule'),
})

trainers = {
    'Oracle': DefaultTrainer,
    'DRF-KDE': DefaultTrainer,
    'MQF2': get_lightning_trainer,
    'Mixture': get_lightning_trainer,
    'Glow': DefaultTrainer,
    'Quantile': get_lightning_trainer,
}
