from .busi_dataset import BUSIDataset
from .busbra_dataset import BUSBRADataset
from .bus_dataset import BUSDataset


def get_dataset(name, root_dir, split='train', fold=0, n_folds=5,
                img_size=224, transform=None):
    """Factory function to get dataset by name."""
    name = name.lower()
    if name == 'busi':
        return BUSIDataset(root_dir, split=split, fold=fold, n_folds=n_folds,
                           img_size=img_size, transform=transform)
    elif name == 'busbra':
        return BUSBRADataset(root_dir, split=split, fold=fold,
                             img_size=img_size, transform=transform)
    elif name == 'bus':
        return BUSDataset(root_dir, split=split, fold=fold, n_folds=n_folds,
                          img_size=img_size, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: busi, busbra, bus")
