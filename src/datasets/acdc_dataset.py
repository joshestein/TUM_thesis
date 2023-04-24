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
        train: bool = True,
        transform: monai.transforms.Compose = None,
        full_volume: bool = False,
        percentage_data: float = 1.0,
    ):
        """
        :param data_dir: Root data dir, in which "training" and "testing" folders are expected
        :param train: Flag used to read train/test data. If `True`, reads training data, otherwise reads testing data.
        :param transform: Any transforms that should be applied
        :param full_volume: Whether to read the full data volume, in addition to the end diastole and systole frames
        :param percentage_data: The fraction of the data to use
        """
        self.data_dir = Path(data_dir)
        self.data_dir = self.data_dir / "training" if train else self.data_dir / "testing"
        self.patients = sorted([f.path for f in os.scandir(self.data_dir) if f.is_dir()])
        self.patients = self.patients[: int(len(self.patients) * percentage_data)]
        self.transform = transform
        self.full_volume = full_volume

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = Path(self.patients[idx])
        patient = patient_dir.name

        config = np.loadtxt(patient_dir / "Info.cfg", dtype=str, delimiter=":")
        end_diastole = int(config[0, 1])
        end_systole = int(config[1, 1])

        randomized_phase = np.random.choice(["end_diastole", "end_systole"])

        if randomized_phase == "end_diastole":
            image = nib.load(patient_dir / f"{patient}_frame{end_diastole:02d}.nii.gz")
            label = nib.load(patient_dir / f"{patient}_frame{end_diastole:02d}_gt.nii.gz")
        else:
            image = nib.load(patient_dir / f"{patient}_frame{end_systole:02d}.nii.gz")
            label = nib.load(patient_dir / f"{patient}_frame{end_systole:02d}_gt.nii.gz")

        sample = {"image": image.get_fdata(dtype=np.float32), "label": label.get_fdata(dtype=np.float32)}

        if self.full_volume:
            original_volume = nib.load(patient_dir / f"{patient}_4d.nii.gz")
            original_volume = original_volume.get_fdata(dtype=np.float32)

            # height, width, slices, time_steps = original_volume.shape
            # num_resampled_slices = int(slices * self.percentage_slices)
            #
            # full_volume = np.ndarray((height, width, num_resampled_slices, time_steps), dtype=np.float32)
            #
            # # Sample slices for each volume at each time step
            # for time_step in range(original_volume.shape[-1]):
            #     full_volume[..., time_step] = remove_slices(original_volume[..., time_step], self.percentage_slices)

            sample["full_volume"] = original_volume

        if self.transform:
            sample = self.transform(sample)

        return sample
