import csv
import os
from pathlib import Path

import monai.transforms
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class MNMsDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        transform: monai.transforms.Compose = None,
        num_training_cases: int | None = None,
        shuffle=True,
    ):
        """
        :param data_dir: Path to training/testing data
        :param transform: Any transforms that should be applied
        :param num_training_cases: The number of cases to use for training
        :param shuffle: Whether to shuffle the dataset
        """
        self.data_dir = data_dir
        self.patients = sorted([Path(f.path) for f in os.scandir(self.data_dir) if f.is_dir()])
        if shuffle:
            np.random.shuffle(self.patients)
        if num_training_cases is not None:
            self.patients = self.patients[:num_training_cases]

        self.transform = transform
        self.cardiac_phase_indexes = self._get_cardiac_phase_indexes()

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
        patient_dir = self.patients[index]
        patient = patient_dir.name

        image = nib.load(patient_dir / f"{patient}_sa.nii.gz")
        label = nib.load(patient_dir / f"{patient}_sa_gt.nii.gz")
        image = image.get_fdata(dtype=np.float32)
        label = label.get_fdata(dtype=np.float32)

        randomized_phase = np.random.choice(["end_diastole", "end_systole"])

        image = image[..., self.cardiac_phase_indexes[patient][randomized_phase]]
        label = label[..., self.cardiac_phase_indexes[patient][randomized_phase]]

        sample = {"image": image, "label": label, "patient": patient}

        if self.transform:
            sample = self.transform(sample)

        return sample
