import os
import tomllib
from pathlib import Path

import cv2
import numpy as np
import torch
from monai.utils import set_determinism
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader

from src.datasets.dataset_helper import DatasetHelperFactory
from src.sam.sam_utils import (
    calculate_dice_for_classes,
    get_bounding_box,
    get_predictions,
    save_figure,
)
from src.utils import setup_dirs


def setup_sam(root_dir: Path, device: str | torch.device, checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    checkpoint = root_dir / "models" / checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam = sam.to(device)
    return sam


def run_inference(
    test_loader: DataLoader, predictor: SamPredictor, device: str | torch.device, out_dir: Path, num_classes=4
):
    """Expects the dataloader to have a batch size of 1."""
    dice_scores = []
    for batch_index, batch in enumerate(test_loader):
        inputs, labels = batch["image"][0].to(device), batch["label"][0].to(device, dtype=torch.uint8)
        inputs = cv2.cvtColor(inputs.permute(2, 1, 0).detach().cpu().numpy(), cv2.COLOR_GRAY2RGB)
        # Scale to 0-255, convert to uint8
        inputs = ((inputs - inputs.min()) * (1 / (inputs.max() - inputs.min()) * 255)).astype("uint8")
        predictor.set_image(inputs)

        labels = labels[0].permute(1, 0)  # Swap W, H

        labels_per_class, bboxes, masks = [], [], []
        for class_index in range(num_classes):
            # Get bounding box for each class of one-hot encoded mask
            label = (labels == class_index).astype(int)
            labels_per_class.append(label)

            bbox = None if np.count_nonzero(label) == 0 else np.array(get_bounding_box(label))
            bboxes.append(bbox)
            mask, _, _ = predictor.predict(box=bbox, multimask_output=False)
            masks.append(mask)

        save_figure(batch_index, inputs, bboxes, labels_per_class, masks, out_dir, num_classes=num_classes)
        dice_scores.append(calculate_dice_for_classes(masks, labels_per_class, num_classes=num_classes))

    return torch.tensor(dice_scores)


def run_batch_inference(test_loader: DataLoader, sam: Sam, device: str | torch.device, out_dir: Path, num_classes=4):
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dice_scores = []
    for batch_index, batch in enumerate(test_loader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device, dtype=torch.uint8)

        masks, labels, boxes, transformed_images = get_predictions(
            sam=sam, transform=resize_transform, inputs=inputs, labels=labels, num_classes=num_classes
        )

        for i in range(inputs.shape[0]):
            save_figure(
                batch_index,
                transformed_images[i],
                boxes[i],
                labels[i],
                masks[i],
                out_dir=out_dir,
                num_classes=num_classes,
            )
            dice_scores.append(calculate_dice_for_classes(masks[i], labels[i], num_classes=num_classes))

        break

    return torch.tensor(dice_scores)


def main(dataset: str):
    root_dir = Path(os.getcwd()).parent.parent
    data_dir, log_dir, out_dir = setup_dirs(root_dir)

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
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    figure_dir = out_dir / "sam" / "figures" / dataset
    os.makedirs(figure_dir, exist_ok=True)
    dice_scores = run_inference(test_loader, SamPredictor(sam), device, figure_dir)
    mean_fg_dice = torch.mean(dice_scores, dim=0)
    print(f"Mean foreground dice: {mean_fg_dice}")
    print(f"Mean dice: {torch.mean(mean_fg_dice)}")


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
