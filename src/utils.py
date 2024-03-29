import os
from pathlib import Path

import numpy as np
import torch
from monai.optimizers import LearningRateFinder
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from src.datasets.acdc_dataset import ACDCDataset
from src.datasets.mnms_dataset import MNMsDataset


def get_train_dataloaders(
    train_dataset: ACDCDataset | MNMsDataset,
    val_dataset: ACDCDataset | MNMsDataset,
    num_training_cases: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    validation_split: float = 0.8,
    shuffle=True,
):
    total_training_number = len(train_dataset)
    train_size = num_training_cases if num_training_cases is not None else int(validation_split * total_training_number)

    # Always use a val_size relative to the total number of samples, not the (limited) number of samples used for
    # training
    val_size = total_training_number - int(validation_split * total_training_number)
    indices = np.arange(total_training_number)

    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:train_size], indices[-val_size:]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, val_loader


def find_optimal_learning_rate(
    model: torch.nn.Module,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    device: str | torch.device,
    train_loader: DataLoader,
    learning_rate: float,
    iterations: int,
    image_key: str = "image",
    label_key: str = "label",
    val_loader: DataLoader | None = None,
):
    lr_finder = LearningRateFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader=train_loader,
        val_loader=val_loader,
        start_lr=learning_rate / 1000,
        end_lr=learning_rate * 1000,
        num_iter=iterations,
        image_extractor=lambda x: x[image_key],
        label_extractor=lambda x: x[label_key],
    )
    optimal_learning_rate, _ = lr_finder.get_steepest_gradient()

    if optimal_learning_rate is None:
        print(f"Optimal learning rate not found, using default learning rate {learning_rate}.")
        optimal_learning_rate = learning_rate
    else:
        print(f"Optimal learning rate found: {optimal_learning_rate}")

    return optimal_learning_rate


def setup_dirs(root_dir: Path):
    # Always prefer external storage to internal storage
    if os.path.exists("/vol/root"):
        root_dir = Path("/vol/root")

    data_dir = root_dir / "data"
    log_dir = root_dir / "logs"
    out_dir = root_dir / "out"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    return data_dir, log_dir, out_dir
