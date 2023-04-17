import gc
import os
import tomllib
from pathlib import Path
from pprint import pprint

import torch
import wandb
from monai.losses import DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.utils import set_determinism

from src.datasets.acdc_dataset import ACDCDataset
from src.metrics import METRICS
from src.train import train
from src.utils import find_optimal_learning_rate, get_train_dataloaders, setup_dirs
from src.transforms.transforms import get_transforms
from src.utils import get_train_dataloaders, setup_dirs


def main():
    root_dir = Path(os.getcwd())
    data_dir, log_dir, root_out_dir = setup_dirs(root_dir)
    data_dir = data_dir / "ACDC" / "database"

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    pprint(config)

    augment = config["hyperparameters"].get("augment", True)
    batch_size = config["hyperparameters"].get("batch_size", 4)
    epochs = config["hyperparameters"].get("epochs", 100)
    learning_rate = config["hyperparameters"].get("learning_rate", 1e-5)
    percentage_data = config["hyperparameters"].get("percentage_data", 1.0)
    validation_split = config["hyperparameters"].get("validation_split", 0.8)

    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    # TODO: weight decay check
    optimizer = torch.optim.Adam(model.parameters())
    optimal_learning_rate = find_optimal_learning_rate(
        model=model,
        optimizer=optimizer,
        criterion=loss_function,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        iterations=100,
    )

    for group in optimizer.param_groups:
        group["lr"] = optimal_learning_rate

    config["hyperparameters"]["optimal_learning_rate"] = optimal_learning_rate

    val_interval = 5

    for percentage_data in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:
        config["hyperparameters"]["percentage_data"] = percentage_data

        for percentage_slices in [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:
            config["hyperparameters"]["percentage_slices"] = percentage_slices

            train_transforms = get_transforms(augment, percentage_slices=percentage_slices)
            train_data = ACDCDataset(
                data_dir=data_dir,
                train=True,
                transform=train_transforms,
                percentage_data=percentage_data,
            )

            train_loader, val_loader = get_train_dataloaders(
                dataset=train_data,
                batch_size=batch_size,
                validation_split=validation_split,
            )

            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=4,
                # channels=(26, 52, 104, 208, 416),
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                norm=Norm.INSTANCE,
                # num_res_units=4,
                # dropout=0.5,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=optimal_learning_rate)

            wandb.init(
                project="acdc-3D-UNet-baseline-restart",
                config=config["hyperparameters"],
                dir=log_dir,
                tags=("limited_data", "limited_slices"),
                reinit=True,
            )
            wandb.config.dataset = "ACDC"
            wandb.config.architecture = "UNet"

            out_dir = root_out_dir / f"percentage_data_{percentage_data}" / f"percentage_slices_{percentage_slices}"
            os.makedirs(out_dir, exist_ok=True)

            epoch_loss_values, metric_values = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                val_interval=val_interval,
                epochs=epochs,
                metrics=METRICS,
                device=device,
                out_dir=out_dir,
            )

            wandb.finish()
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()