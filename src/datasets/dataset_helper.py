from abc import ABC, abstractmethod
from pathlib import Path

from src.datasets.acdc_dataset import ACDCDataset
from src.datasets.mnms_dataset import MNMsDataset
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
        num_training_cases: int | None = None,
        num_slices: int | None = None,
        sample_regions: list[str] = ("apex", "mid", "base"),
    ):
        self.data_dir = data_dir
        self.num_training_cases = num_training_cases
        self.random_slice = spatial_dims == 2

        self.train_transforms, self.val_transforms = get_transforms(
            spatial_dims=spatial_dims,
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
            num_training_cases=self.num_training_cases,
            random_slice=self.random_slice,
        )
        val_data = ACDCDataset(
            data_dir=data_dir,
            transform=self.val_transforms,
            num_training_cases=self.num_training_cases,
            random_slice=self.random_slice,
        )

        return train_data, val_data

    def get_test_dataset(self):
        test_data = ACDCDataset(
            data_dir=self.data_dir / "testing",
            transform=self.val_transforms,
            random_slice=self.random_slice,
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
            num_training_cases=self.num_training_cases,
            random_slice=self.random_slice,
        )
        val_data = MNMsDataset(
            data_dir=data_dir,
            transform=self.val_transforms,
            num_training_cases=self.num_training_cases,
            random_slice=self.random_slice,
        )

        return train_data, val_data

    def get_test_dataset(self):
        test_data = MNMsDataset(
            data_dir=self.data_dir / "Testing",
            transform=self.val_transforms,
            random_slice=self.random_slice,
        )

        return test_data
