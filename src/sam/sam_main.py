import os
import tomllib
from pathlib import Path
from pprint import pprint

import torch
import wandb
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.datasets.dataset_helper import DatasetHelperFactory
from src.metrics import MetricHandler
from src.sam.sam_inference import setup_sam
from src.sam.sam_train import train
from src.utils import get_train_dataloaders, setup_dirs


def main(
    dataset_name: str,
    pos_sample_points: int,
    neg_sample_points: int = 0,
    use_bboxes: bool = True,
    num_training_cases: int | None = None,
    num_slices: int | None = None,
):
    root_dir = Path(os.getcwd())
    data_dir, log_dir, root_out_dir = setup_dirs(root_dir)

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    pprint(config)
    augment = config["hyperparameters"].get("augment", True)
    batch_size = config["hyperparameters"].get("batch_size", 4)
    epochs = config["hyperparameters"].get("epochs", 100)
    learning_rate = 1e-4
    spatial_dims = 2
    validation_split = config["hyperparameters"].get("validation_split", 0.8)

    dataset_helper = DatasetHelperFactory(dataset_name=dataset_name)
    dataset = dataset_helper.dataset

    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_function = DiceCELoss(to_onehot_y=False, softmax=False, sigmoid=True, batch=True)

    train_data, val_data = dataset(
        spatial_dims=spatial_dims,
        data_dir=data_dir,
        augment=augment,
        num_training_cases=num_training_cases,
        num_slices=num_slices,
        nnunet_transforms=True,
        force_foreground_classes=True,
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
    neg_samples_str = "" if neg_sample_points == 0 else f"neg_samples_{neg_sample_points}"
    num_training_cases_str = "" if num_training_cases is None else f"num_training_cases_{num_training_cases}"
    num_slices_str = "" if num_slices is None else f"num_slices_{num_slices}"

    out_dir = (
        root_out_dir
        / "sam"
        / dataset_name
        / "_".join(filter(None, (num_training_cases_str, num_slices_str)))
        / num_samples_str
    )
    if neg_samples_str:
        out_dir = out_dir / neg_samples_str

    os.makedirs(out_dir, exist_ok=True)

    wandb.init(
        project=f"sam_{dataset_name}",
        name=f"{'_'.join(filter(None, (num_training_cases_str, num_slices_str, num_samples_str, neg_samples_str)))}",
        config=config["hyperparameters"],
        dir=log_dir,
        mode="disabled",
        reinit=True,
    )
    wandb.config.dataset = dataset_name
    wandb.config.pos_sample_points = pos_sample_points
    wandb.config.neg_sample_points = neg_sample_points

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
        neg_sample_points=neg_sample_points,
        use_bboxes=use_bboxes,
        metric_handler=metric_handler,
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
    parser.add_argument("--bboxes", "-b", type=bool, default=True)
    parser.add_argument(
        "--num_training_cases",
        "-c",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--num_slices",
        "-s",
        type=int,
        required=False,
    )
    args = parser.parse_args()

    dataset_case_dict = {"acdc": [8, 24, 32, 48, 80, 160], "mnms": [8, 24, 32, 48, 80, 160, 192, 240]}
    dataset_slice_dict = {
        "acdc": [1, 2, 4, 5, 6, 8, 10, 13, 14, 16, 20],
        "mnms": [1, 2, 4, 5, 6, 8, 10, 13, 14],
    }

    for dataset in ["acdc", "mnms"]:
        for num_cases in dataset_case_dict[dataset]:
            main(
                dataset_name=dataset,
                pos_sample_points=args.pos_sample_points,
                neg_sample_points=args.neg_sample_points,
                use_bboxes=args.bboxes,
                num_training_cases=num_cases,
            )
