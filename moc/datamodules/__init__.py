from .real_datamodule import RealDataModule
from .toy_datamodule import ToyDataModule
from moc.configs.datasets import toy_dataset_groups, real_dataset_groups


def get_datamodule(group):
    from moc.configs.ts_datasets import is_ts_group

    if group in toy_dataset_groups:
        return ToyDataModule
    elif group in real_dataset_groups:
        return RealDataModule
    elif is_ts_group(group):
        from .timeseries_datamodule import TimeSeriesDataModule
        return TimeSeriesDataModule
    elif group == 'cifar10':
        # Imported lazily: torchvision is only needed for this one group, and requiring it would
        # make the time-series path depend on it too.
        from .cifar10_datamodule import CIFAR10DataModule
        return CIFAR10DataModule
    raise ValueError(f'Unknown datamodule {group}')


def load_datamodule(rc):
    datamodule_cls = get_datamodule(rc.dataset_group)
    return datamodule_cls(
        rc=rc,
        seed=2000 + rc.run_id,
    )
