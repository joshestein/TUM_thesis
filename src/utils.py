import os
from pathlib import Path

import torch
from monai.optimizers import LearningRateFinder
from torch.optim import Optimizer
from torch.utils.data import DataLoader


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
):
    lr_finder = LearningRateFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        start_lr=int(learning_rate / 1000),
        end_lr=int(learning_rate * 1000),
        num_iter=iterations,
        image_extractor=lambda x: x[image_key][..., 0] if model.dimensions == 2 else x[image_key],
        label_extractor=lambda x: x[label_key][..., 0] if model.dimensions == 2 else x[label_key],
    )
    optimal_learning_rate, _ = lr_finder.get_steepest_gradient()

    if optimal_learning_rate is None:
        print(f"Optimal learning rate not found, using default learning rate {learning_rate}.")
        optimal_learning_rate = learning_rate
    else:
        print(f"Optimal learning rate found: {optimal_learning_rate}")

    return optimal_learning_rate


def setup_dirs(root_dir=Path(os.getcwd())):
    data_dir = root_dir / "data"
    log_dir = root_dir / "logs"
    out_dir = root_dir / "out"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    return data_dir, log_dir, out_dir
