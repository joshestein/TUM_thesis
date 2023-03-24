import os
from pathlib import Path

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
    def __init__(self, data_dir: str | Path, train=True, transform=None, full_volume=False, percentage_data=1.0):
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
        config = np.loadtxt(patient_dir / "Info.cfg", dtype=str, delimiter=":")

        end_diastole = int(config[0, 1])
        end_systole = int(config[1, 1])

        patient = patient_dir.name
        end_diastole_image = nib.load(patient_dir / f"{patient}_frame{end_diastole:02d}.nii.gz")
        end_diastole_label = nib.load(patient_dir / f"{patient}_frame{end_diastole:02d}_gt.nii.gz")
        end_systole_image = nib.load(patient_dir / f"{patient}_frame{end_systole:02d}.nii.gz")
        end_systole_label = nib.load(patient_dir / f"{patient}_frame{end_systole:02d}_gt.nii.gz")

        end_diastole_image = end_diastole_image.get_fdata(dtype=np.float32)
        end_diastole_label = end_diastole_label.get_fdata(dtype=np.float32)
        end_systole_image = end_systole_image.get_fdata(dtype=np.float32)
        end_systole_label = end_systole_label.get_fdata(dtype=np.float32)

        sample = {
            "end_diastole": end_diastole_image,
            "end_diastole_label": end_diastole_label,
            "end_systole": end_systole_image,
            "end_systole_label": end_systole_label,
        }

        if self.full_volume:
            full_volume = nib.load(patient_dir / f"{patient}_4d.nii.gz")
            full_volume = full_volume.get_fdata(dtype=np.float32)
            sample["full_volume"] = full_volume

        if self.transform:
            sample = self.transform(sample)

        return sample
