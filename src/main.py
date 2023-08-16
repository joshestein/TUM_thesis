import gc
import os
import tomllib
from pathlib import Path
from pprint import pprint

import torch
import wandb
from monai.losses import DiceCELoss, DiceLoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.utils import set_determinism

from src.datasets.dataset_helper import DatasetHelperFactory
from src.train import train
from src.utils import find_optimal_learning_rate, get_train_dataloaders, setup_dirs


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
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, batch=True)
    elif loss == "dice_ce":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, batch=True)
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

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=4,
        # channels=(26, 52, 104, 208, 416),
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        norm=Norm.BATCH,
        num_res_units=4,
        dropout=0.5,
    ).to(device)
    torch.save(model, root_dir / "initial_model.pt")

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

            train_data, val_data = dataset(
                spatial_dims=spatial_dims,
                data_dir=data_dir,
                augment=augment,
                percentage_slices=percentage_slices,
                percentage_data=percentage_data,
            ).get_training_datasets()

            train_loader, val_loader = get_train_dataloaders(
                train_dataset=train_data,
                val_dataset=val_data,
                batch_size=batch_size,
                validation_split=validation_split,
            )

            model = torch.load(root_dir / "initial_model.pt")
            optimizer = torch.optim.Adam(model.parameters(), lr=optimal_learning_rate)

            wandb.init(
                project=f"{dataset_name}-{spatial_dims}D-UNet-fresh-restart",
                config=config["hyperparameters"],
                dir=log_dir,
                tags=("limited_data", "limited_slices"),
                reinit=True,
            )
            wandb.config.dataset = dataset_name
            wandb.config.architecture = "UNet"

            out_dir = (
                root_out_dir
                / f"UNet_{spatial_dims}D"
                / f"{dataset_name}"
                / f"percentage_data_{percentage_data}"
                / f"percentage_slices_{percentage_slices}"
            )
            os.makedirs(out_dir, exist_ok=True)

            _, _ = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                val_interval=val_interval,
                epochs=epochs,
                device=device,
                out_dir=out_dir,
            )

            wandb.finish()
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="acdc",
    )
    args = parser.parse_args()

    main(args.dataset)
