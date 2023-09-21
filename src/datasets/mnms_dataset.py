import csv
import os
from pathlib import Path

import batchgenerators.transforms.abstract_transforms
import monai.transforms
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class MNMsDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        transform: monai.transforms.Compose | batchgenerators.transforms.abstract_transforms.Compose = None,
        num_training_cases: int | None = None,
        shuffle=True,
        random_slice=False,
    ):
        """
        :param data_dir: Path to training/testing data
        :param transform: Any transforms that should be applied
        :param num_training_cases: The number of cases to use for training
        :param shuffle: Whether to shuffle the dataset
        """
        self.data_dir = data_dir
        patient_dirs = sorted([Path(f.path) for f in os.scandir(self.data_dir) if f.is_dir()])
        if shuffle:
            np.random.shuffle(patient_dirs)

        self.patients, self.labels = [], []
        for patient in patient_dirs:
            self.patients.extend(sorted(patient.glob("*[0-9].nii.gz")))
            self.labels.extend(sorted(patient.glob("*[0-9]_gt.nii.gz")))

        if num_training_cases is not None:
            self.patients = self.patients[:num_training_cases]
            self.labels = self.labels[:num_training_cases]

        self.transform = transform
        self.random_slice = random_slice

    def _get_cardiac_phase_indexes(self):
        """Reads the CSV metadata file to extract the end diastole and end systole frames for each volume.

        :returns A nested dictionary, where outer keys are patient folder names and inner keys are "end_diastole" and
        "end_systole", i.e. {"patient_name": {"end_diastole": 0, "end_systole": 12}}
        """
        csv_file = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
        root_dir = self._get_root_dir(csv_file)

        cardiac_phase_indexes = {}
        with open(root_dir / csv_file) as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            patient_index = headers.index("External code")
            ed_index = headers.index("ED")
            es_index = headers.index("ES")
            for row in reader:
                cardiac_phase_indexes[row[patient_index]] = {
                    "end_diastole": int(row[ed_index]),
                    "end_systole": int(row[es_index]),
                }

        return cardiac_phase_indexes

    def _get_root_dir(self, csv_file: str):
        """Returns the root directory of the dataset, which contains the CSV metadata file."""
        num_tries = 5
        path = self.data_dir
        for _ in range(num_tries):
            if (path / csv_file).exists():
                return path
            path = path.parent

        return path

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        image = nib.load(self.patients[index])
        label = nib.load(self.labels[index])
        patient = self.patients[index].parent.name

        image = image.get_fdata(dtype=np.float32)
        label = label.get_fdata(dtype=np.float32)

        if self.random_slice:
            slice_index = np.random.randint(0, image.shape[-1])
            # TODO: use slicer and only read random slice into memory
            image_slice = image[..., slice_index]
            label_slice = label[..., slice_index].astype(int)

            label_max = label_slice.max()
            # Ensure we don't sample empty slices
            while label_max == 0:
                slice_index = np.random.randint(0, image.shape[-1])
                image_slice = image[..., slice_index]
                label_slice = label[..., slice_index].astype(int)
                label_max = label_slice.max()

            image = image_slice[np.newaxis, ...]  # Add channel dimension
            # MNMs has 4 classes
            label = np.moveaxis(np.eye(4)[label_slice], -1, 0)  # Convert to onehot, move channel to first dim
            patient = f"{patient}_slice_{slice_index}"

        sample = {
            "image": image,
            "label": label,
            "patient": patient,
        }

        if self.transform and isinstance(self.transform, monai.transforms.Compose):
            sample = self.transform(sample)
        elif self.transform and isinstance(self.transform, batchgenerators.transforms.abstract_transforms.Compose):
            sample = self.transform(**sample)

        return sample
