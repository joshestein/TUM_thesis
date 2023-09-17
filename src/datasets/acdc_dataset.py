import os
from pathlib import Path

import monai.transforms
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        transform: monai.transforms.Compose = None,
        full_volume: bool = False,
        num_training_cases: int | None = None,
    ):
        """
        :param data_dir: Root data dir, in which "training" and "testing" folders are expected
        :param transform: Any transforms that should be applied
        :param full_volume: Whether to read the full data volume, in addition to the end diastole and systole frames
        :param num_training_cases: The number of cases to use for training
        """
        self.data_dir = Path(data_dir)

        patient_dirs = sorted([Path(f.path) for f in os.scandir(self.data_dir) if f.is_dir()])
        self.patients, self.labels = [], []
        for patient_dir in patient_dirs:
            self.patients.extend(sorted(patient_dir.glob("*[0-9].nii.gz")))
            self.labels.extend(sorted(patient_dir.glob("*_gt.nii.gz")))

        if num_training_cases is not None:
            self.patients = self.patients[:num_training_cases]
            self.labels = self.labels[:num_training_cases]

        self.transform = transform
        self.full_volume = full_volume

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        image = nib.load(self.patients[index])
        label = nib.load(self.labels[index])
        patient = self.patients[index].parent.name

        sample = {
            "image": image.get_fdata(dtype=np.float32),
            "label": label.get_fdata(dtype=np.float32),
            "patient": patient,
        }

        # if self.full_volume:
        #     original_volume = nib.load(patient_dir / f"{patient}_4d.nii.gz")
        #     original_volume = original_volume.get_fdata(dtype=np.float32)

        # height, width, slices, time_steps = original_volume.shape
        # num_resampled_slices = int(slices * self.percentage_slices)
        #
        # full_volume = np.ndarray((height, width, num_resampled_slices, time_steps), dtype=np.float32)
        #
        # # Sample slices for each volume at each time step
        # for time_step in range(original_volume.shape[-1]):
        #     full_volume[..., time_step] = remove_slices(original_volume[..., time_step], self.percentage_slices)

        # sample["full_volume"] = original_volume

        if self.transform:
            sample = self.transform(sample)

        return sample
