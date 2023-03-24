import os
from pathlib import Path

import monai.transforms
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


def sample_slices(volume: np.ndarray, percentage_slices: float):
    slices = volume.shape[-1]
    num_slices = int(slices * percentage_slices)
    start_slice = slices // 2 - num_slices // 2
    end_slice = start_slice + num_slices
    return volume[..., start_slice:end_slice]


class ACDCDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        train: bool = True,
        transform: monai.transforms.Compose = None,
        full_volume: bool = False,
        percentage_data: float = 1.0,
        percentage_slices: float = 1.0,
    ):
        """
        :param data_dir: Root data dir, in which "training" and "testing" folders are expected
        :param train: Flag used to read train/test data. If `True`, reads training data, otherwise reads testing data.
        :param transform: Any transforms that should be applied
        :param full_volume: Whether to read the full data volume, in addition to the end diastole and systole frames
        :param percentage_data: The fraction of the data to use
        :param percentage_slices: The fraction of slices to extract from a given 3D volume. The slices are extracted around
            the center of the volume. For example, if `percentage_slices=0.5`, then the slices will be from the first quarter
            to third quarter of the volume.
        """
        self.data_dir = Path(data_dir)
        self.data_dir = self.data_dir / "training" if train else self.data_dir / "testing"
        self.patients = sorted([f.path for f in os.scandir(self.data_dir) if f.is_dir()])
        self.patients = self.patients[: int(len(self.patients) * percentage_data)]
        self.transform = transform
        self.full_volume = full_volume
        self.percentage_slices = percentage_slices

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = Path(self.patients[idx])
        config = np.loadtxt(patient_dir / "Info.cfg", dtype=str, delimiter=":")

        end_diastole = int(config[0, 1])
        end_systole = int(config[1, 1])

        patient = patient_dir.name

        sample = {
            "end_diastole": nib.load(patient_dir / f"{patient}_frame{end_diastole:02d}.nii.gz"),
            "end_diastole_label": nib.load(patient_dir / f"{patient}_frame{end_diastole:02d}_gt.nii.gz"),
            "end_systole": nib.load(patient_dir / f"{patient}_frame{end_systole:02d}.nii.gz"),
            "end_systole_label": nib.load(patient_dir / f"{patient}_frame{end_systole:02d}_gt.nii.gz"),
        }

        for key, image in sample.items():
            image = image.get_fdata(dtype=np.float32)
            image = sample_slices(image, self.percentage_slices)
            sample[key] = image

        if self.full_volume:
            original_volume = nib.load(patient_dir / f"{patient}_4d.nii.gz")
            original_volume = original_volume.get_fdata(dtype=np.float32)

            height, width, slices, time_steps = original_volume.shape
            num_resampled_slices = int(slices * self.percentage_slices)

            full_volume = np.ndarray((height, width, num_resampled_slices, time_steps), dtype=np.float32)

            # Sample slices for each volume at each time step
            for time_step in range(original_volume.shape[-1]):
                full_volume[..., time_step] = sample_slices(original_volume[..., time_step], self.percentage_slices)

            sample["full_volume"] = full_volume

        if self.transform:
            sample = self.transform(sample)

        return sample
