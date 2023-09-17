import csv
import os
from pathlib import Path

import nibabel as nib


def read_cardiac_phase_info(data_dir: Path):
    csv_file = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

    cardiac_phase_indexes = {}
    with open(data_dir / csv_file) as csvfile:
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


def save_extracted_patient_frames(patients: list[Path], cardiac_phase_info: dict[str, dict[str, int]]):
    for patient in patients:
        image = nib.load(patient / f"{patient.name}_sa.nii.gz")
        label = nib.load(patient / f"{patient.name}_sa_gt.nii.gz")

        ed_frame = cardiac_phase_info[patient.name]["end_diastole"]
        es_frame = cardiac_phase_info[patient.name]["end_systole"]
        end_diastole = image.slicer[..., ed_frame]
        end_systole = image.slicer[..., es_frame]
        label_end_diastole = label.slicer[..., ed_frame]
        label_end_systole = label.slicer[..., es_frame]

        nib.save(end_diastole, patient / f"{patient.name}_frame_{ed_frame:02d}.nii.gz")
        nib.save(end_systole, patient / f"{patient.name}_frame_{es_frame:02d}.nii.gz")
        nib.save(label_end_diastole, patient / f"{patient.name}_frame_{ed_frame:02d}_gt.nii.gz")
        nib.save(label_end_systole, patient / f"{patient.name}_frame_{es_frame:02d}_gt.nii.gz")


def main(data_dir: Path):
    cardiac_phase_info = read_cardiac_phase_info(data_dir)

    train_patients = sorted([Path(f.path) for f in os.scandir(data_dir / "Training" / "Labeled") if f.is_dir()])
    test_patients = sorted([Path(f.path) for f in os.scandir(data_dir / "Testing") if f.is_dir()])

    save_extracted_patient_frames(train_patients, cardiac_phase_info)
    save_extracted_patient_frames(test_patients, cardiac_phase_info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    main(Path(args.data_dir))
