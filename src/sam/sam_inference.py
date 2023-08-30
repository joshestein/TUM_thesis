import json
import os
import tomllib
from pathlib import Path

import numpy as np
import torch
import wandb
from monai.utils import set_determinism
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

from src.datasets.dataset_helper import DatasetHelperFactory
from src.sam.sam_utils import (
    calculate_dice_for_classes,
    convert_to_normalized_grayscale,
    get_batch_predictions,
    get_numpy_bounding_box,
    get_sam_points,
    save_figure,
)
from src.utils import setup_dirs


def setup_sam(root_dir: Path, device: str | torch.device, checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    checkpoint = root_dir / "models" / checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam = sam.to(device)
    return sam


def run_inference(
    test_loader: DataLoader,
    predictor: SamPredictor,
    device: str | torch.device,
    out_dir: Path,
    num_sample_points: int,
    use_bboxes: bool = True,
    use_points: bool = True,
    num_classes=4,
):
    """Expects the dataloader to have a batch size of 1."""
    dice_scores = []
    for batch in test_loader:
        inputs, labels = batch["image"][0].to(device), batch["label"][0].to(device, dtype=torch.uint8)
        patient = batch["patient"][0]

        print(f"Predicting patient: {patient}")
        inputs = convert_to_normalized_grayscale(inputs)
        predictor.set_image(inputs)

        labels = labels[0].permute(1, 0)  # Swap W, H

        bboxes, masks, points, point_labels = [], [], [], []
        labels_per_class = [np.array((labels == class_index).astype(int)) for class_index in range(num_classes)]
        if any(np.count_nonzero(label) == 0 for label in labels_per_class):
            print(f"Skipping {patient} as it contains empty labels.")
            continue

        for label in labels_per_class:
            bbox = get_numpy_bounding_box(label) if use_bboxes else None
            point_coords, point_label = get_sam_points(label, num_sample_points) if use_points else (None, None)
            mask, _, _ = predictor.predict(
                box=bbox, point_coords=point_coords, point_labels=point_label, multimask_output=False
            )
            masks.append(mask)
            bboxes.append(bbox)
            points.append(point_coords)
            point_labels.append(point_label)

        save_figure(
            patient_name=patient,
            inputs=inputs,
            bboxes=bboxes,
            labels=labels_per_class,
            masks=masks,
            out_dir=out_dir,
            num_classes=num_classes,
            points=points,
            point_labels=point_labels,
        )
        dice_scores.append(calculate_dice_for_classes(masks, labels_per_class, num_classes=num_classes))

    return torch.tensor(dice_scores)


def run_batch_inference(
    test_loader: DataLoader, sam: Sam, device: str | torch.device, out_dir: Path, num_sample_points: int, num_classes=4
):
    sam.eval()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dice_scores = []
    for batch_index, batch in enumerate(test_loader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device, dtype=torch.uint8)
        patient = batch["patient"]

        with torch.no_grad():
            masks, labels, boxes, points, transformed_images = get_batch_predictions(
                sam=sam,
                transform=resize_transform,
                inputs=inputs,
                labels=labels,
                num_points=num_sample_points,
                num_classes=num_classes,
            )

        for i in range(inputs.shape[0]):
            save_figure(
                patient,
                transformed_images[i],
                boxes[i],
                labels[i],
                masks[i],
                out_dir=out_dir,
                num_classes=num_classes,
                points=points[i],
                point_labels=np.ones(num_sample_points) * len(points),
            )
            dice_scores.append(calculate_dice_for_classes(masks[i], labels[i], num_classes=num_classes))

    return torch.tensor(dice_scores)


def main(dataset: str, num_sample_points: int, use_bboxes: bool):
    wandb.init(
        project=f"sam_inference",
        name=f"{dataset}_num_samples_{num_sample_points}",
        config={"dataset": dataset, "num_sample_points": num_sample_points},
        mode="disabled",
        reinit=True,
    )

    print(f"Starting inference for {dataset}...")
    root_dir = Path(os.getcwd())

    # External storage should always be prioritized as our data/log/out dir
    if os.path.exists("/vol/root"):
        data_dir, log_dir, out_dir = setup_dirs(Path("/vol/root"))
    else:
        data_dir, log_dir, out_dir = setup_dirs(root_dir)

    out_dir = out_dir / "sam" / f"{dataset}_num_samples_{num_sample_points}"

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    augment = config["hyperparameters"].get("augment", True)
    # batch_size = config["hyperparameters"].get("batch_size", 4)
    batch_size = 1
    spatial_dims = 2
    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sam = setup_sam(root_dir, device)

    dataset_helper = DatasetHelperFactory(dataset_name=dataset)
    dataset_helper = dataset_helper.dataset(spatial_dims=spatial_dims, data_dir=data_dir, augment=augment)

    test_data = dataset_helper.get_test_dataset()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    figure_dir = out_dir / "figures"
    os.makedirs(figure_dir, exist_ok=True)
    dice_scores = run_inference(
        test_loader, SamPredictor(sam), device, figure_dir, use_bboxes=use_bboxes, num_sample_points=num_sample_points
    )
    # dice_scores = run_batch_inference(test_loader, sam, device, figure_dir, num_sample_points=num_sample_points)
    mean_fg_dice = torch.mean(dice_scores, dim=0)
    print(f"Dice per class: {mean_fg_dice}")
    print(f"Mean dice: {torch.mean(mean_fg_dice)}")

    wandb.log({"dice_per_class": mean_fg_dice.tolist()})
    wandb.log({"mean_dice": torch.mean(mean_fg_dice)})

    results = {
        "dice_per_class": mean_fg_dice.tolist(),
        "mean_dice": torch.mean(mean_fg_dice).item(),
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


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
        "--num_sample_points",
        "-s",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--use_bboxes",
        "-b",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    for dataset in ["acdc", "mnms"]:
        for num_points in [2, 3, 5]:
            main(dataset, num_points, args.use_bboxes)
