import os
import tomllib
from pathlib import Path
from pprint import pprint

import torch
from monai.losses import DiceCELoss, DiceLoss
from monai.utils import set_determinism

from src.datasets.dataset_helper import DatasetHelperFactory
from src.sam.sam_inference import setup_sam
from src.utils import get_train_dataloaders, setup_dirs


def main(dataset_name: str):
    root_dir = Path(os.getcwd()).parent
    data_dir, log_dir, root_out_dir = setup_dirs(root_dir)

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    pprint(config)
    augment = config["hyperparameters"].get("augment", True)
    batch_size = config["hyperparameters"].get("batch_size", 4)
    epochs = config["hyperparameters"].get("epochs", 100)
    learning_rate = config["hyperparameters"].get("learning_rate", 1e-5)
    spatial_dims = config["hyperparameters"].get("spatial_dimensions", 3)
    validation_split = config["hyperparameters"].get("validation_split", 0.8)

    dataset_helper = DatasetHelperFactory(dataset_name=dataset_name)
    dataset = dataset_helper.dataset

    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss = "dice"
    if loss == "dice":
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    elif loss == "dice_ce":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        return

    train_data, val_data = dataset(
        spatial_dims=spatial_dims, data_dir=data_dir, augment=augment, percentage_slices=1.0, percentage_data=1.0
    ).get_training_datasets()

    train_loader, val_loader = get_train_dataloaders(
        train_dataset=train_data,
        val_dataset=val_data,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    sam = setup_sam(root_dir, device)


if __name__ == "__main__":
    main()
