import os
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    def __init__(self, data_dir: str | Path, train=True, transform=None):
        self.data_dir = Path(data_dir)
        self.data_dir = (
            self.data_dir / "training" if train else self.data_dir / "testing"
        )
        self.patients = sorted(
            [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        )
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = Path(self.patients[idx])
        config = np.loadtxt(patient_dir / "Info.cfg", dtype=str, delimiter=":")

        end_diastole = int(config[0, 1])
        end_systole = int(config[1, 1])

        patient = patient_dir.name
        end_diastole_image = nib.load(
            patient_dir / f"{patient}_frame{end_diastole:02d}.nii.gz"
        )
        end_diastole_label = nib.load(
            patient_dir / f"{patient}_frame{end_diastole:02d}_gt.nii.gz"
        )
        end_systole_image = nib.load(
            patient_dir / f"{patient}_frame{end_systole:02d}.nii.gz"
        )
        end_systole_label = nib.load(
            patient_dir / f"{patient}_frame{end_systole:02d}_gt.nii.gz"
        )

        end_diastole_image = end_diastole_image.get_fdata(dtype=np.float32)
        end_diastole_label = end_diastole_label.get_fdata(dtype=np.float32)
        end_systole_image = end_systole_image.get_fdata(dtype=np.float32)
        end_systole_label = end_systole_label.get_fdata(dtype=np.float32)

        sample = {
            "end_diastole": end_diastole_image,
            "end_diastole_label": end_diastole_label
            # "end_systole": np.expand_dims(end_systole_image, axis=0),
            # "end_systole_label": np.expand_dims(end_systole_label, axis=0),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
