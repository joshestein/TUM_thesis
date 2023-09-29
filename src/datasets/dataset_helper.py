from abc import ABC, abstractmethod
from pathlib import Path

from src.datasets.acdc_dataset import ACDCDataset
from src.datasets.mnms_dataset import MNMsDataset
from src.transforms.nnunet_transforms import get_nnunet_transforms
from src.transforms.transforms import get_transforms


class DatasetHelperFactory:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @property
    def dataset(self):
        if self.dataset_name == "acdc":
            return ACDCDatasetHelper
        elif self.dataset_name == "mnms":
            return MNMsDatasetHelper
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")


class DatasetHelper(ABC):
    def __init__(
        self,
        spatial_dims: int,
        data_dir: Path,
        augment: bool = True,
        num_slices: int | None = None,
        sample_regions: list[str] = ("apex", "mid", "base"),
        nnunet_transforms=False,
        force_foreground_classes: bool = False,
    ):
        self.data_dir = data_dir
        self.random_slice = spatial_dims == 2
        self.force_all_foreground_classes = force_foreground_classes

        if nnunet_transforms:
            self.train_transforms, self.val_transforms = get_nnunet_transforms()
        else:
            self.train_transforms, self.val_transforms = get_transforms(
                augment=augment,
                num_slices=num_slices,
                sample_regions=sample_regions,
            )

    @abstractmethod
    def get_training_datasets(self):
        ...

    @abstractmethod
    def get_test_dataset(self):
        ...


class ACDCDatasetHelper(DatasetHelper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_dir / "ACDC" / "database"

    def get_training_datasets(self):
        data_dir = self.data_dir / "training"
        train_data = ACDCDataset(
            data_dir=data_dir,
            transform=self.train_transforms,
            random_slice=self.random_slice,
            force_foreground=self.force_all_foreground_classes,
        )
        val_data = ACDCDataset(
            data_dir=data_dir,
            transform=self.val_transforms,
            random_slice=self.random_slice,
            force_foreground=self.force_all_foreground_classes,
        )

        return train_data, val_data

    def get_test_dataset(self):
        test_data = ACDCDataset(
            data_dir=self.data_dir / "testing",
            transform=self.val_transforms,
            random_slice=self.random_slice,
            force_foreground=self.force_all_foreground_classes,
        )

        return test_data


class MNMsDatasetHelper(DatasetHelper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = self.data_dir / "MNMs"

    def get_training_datasets(self):
        data_dir = self.data_dir / "Training" / "Labeled"
        train_data = MNMsDataset(
            data_dir=data_dir,
            transform=self.train_transforms,
            random_slice=self.random_slice,
            force_foreground=self.force_all_foreground_classes,
        )
        val_data = MNMsDataset(
            data_dir=data_dir,
            transform=self.val_transforms,
            random_slice=self.random_slice,
            force_foreground=self.force_all_foreground_classes,
        )

        return train_data, val_data

    def get_test_dataset(self):
        test_data = MNMsDataset(
            data_dir=self.data_dir / "Testing",
            transform=self.val_transforms,
            random_slice=self.random_slice,
            force_foreground=self.force_all_foreground_classes,
        )

        return test_data
