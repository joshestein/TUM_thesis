import os
import tomllib
from pathlib import Path
from pprint import pprint

import torch
import wandb
from monai.losses import DiceCELoss, DiceLoss
from monai.utils import set_determinism

from src.datasets.dataset_helper import DatasetHelperFactory
from src.metrics import MetricHandler
from src.sam.sam_inference import setup_sam
from src.sam.sam_train import train
from src.utils import get_train_dataloaders, setup_dirs


def main(
    dataset_name: str,
    pos_sample_points: int,
    neg_sample_points: int | None = None,
    num_training_cases: int | None = None,
):
    root_dir = Path(os.getcwd())
    data_dir, log_dir, root_out_dir = setup_dirs(root_dir)

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    pprint(config)
    augment = config["hyperparameters"].get("augment", True)
    batch_size = config["hyperparameters"].get("batch_size", 4)
    epochs = config["hyperparameters"].get("epochs", 100)
    learning_rate = config["hyperparameters"].get("learning_rate", 1e-5)
    loss = config["hyperparameters"].get("loss", "dice")
    spatial_dims = config["hyperparameters"].get("spatial_dimensions", 3)
    validation_split = config["hyperparameters"].get("validation_split", 0.8)

    dataset_helper = DatasetHelperFactory(dataset_name=dataset_name)
    dataset = dataset_helper.dataset

    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if loss == "dice":
        loss_function = DiceLoss(to_onehot_y=False, softmax=False, batch=True)
    elif loss == "dice_ce":
        loss_function = DiceCELoss(to_onehot_y=False, softmax=False, batch=True)
    else:
        return

    train_data, val_data = dataset(
        spatial_dims=spatial_dims,
        data_dir=data_dir,
        augment=augment,
        num_training_cases=num_training_cases,
    ).get_training_datasets()

    train_loader, val_loader = get_train_dataloaders(
        train_dataset=train_data,
        val_dataset=val_data,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    sam = setup_sam(root_dir, device)
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=learning_rate)

    num_samples_str = f"num_samples_{pos_sample_points}"
    neg_samples_str = "" if neg_sample_points is None else f"neg_samples_{neg_sample_points}"

    out_dir = root_out_dir / "sam" / dataset_name / f"num_training_cases_{num_training_cases}" / num_samples_str
    if neg_sample_points is not None:
        out_dir = out_dir / neg_samples_str

    os.makedirs(out_dir, exist_ok=True)

    wandb.init(
        project=f"sam_{dataset_name}",
        name=f"{'_'.join(filter(None, (num_samples_str, neg_samples_str)))}",
        config=config["hyperparameters"],
        dir=log_dir,
        mode="disabled",
        reinit=True,
    )
    wandb.config.dataset = dataset_name

    metric_handler = MetricHandler("dice")

    train(
        sam=sam,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        val_interval=5,
        epochs=epochs,
        device=device,
        out_dir=out_dir,
        pos_sample_points=pos_sample_points,
        metric_handler=metric_handler,
        neg_sample_points=neg_sample_points,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="acdc",
    )
    parser.add_argument(
        "--pos_sample_points",
        "-p",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--neg_sample_points",
        "-n",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_training_cases",
        "-c",
        type=int,
        required=False,
    )
    args = parser.parse_args()

    for dataset in ["27", "114"]:
        for num_training_cases in [8, 24, 32, 48, 80, 160, 192, 240]:
            main(dataset, args.pos_sample_points, args.neg_sample_points, num_training_cases)
