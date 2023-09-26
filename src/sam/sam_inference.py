import json
import os
import tomllib
from pathlib import Path

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
    calculate_hd_for_classes,
    convert_to_normalized_colour,
    get_batch_predictions,
    get_numpy_bounding_box,
    get_sam_points,
    save_figure,
)
from src.utils import setup_dirs


def setup_sam(root_dir: Path, device: str | torch.device, checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    checkpoint = root_dir / checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam = sam.to(device)
    return sam


def run_inference(
    test_loader: DataLoader,
    predictor: SamPredictor,
    device: str | torch.device,
    out_dir: Path,
    pos_sample_points: int,
    neg_sample_points: int = 0,
    use_bboxes: bool = True,
    use_points: bool = True,
):
    """Expects the dataloader to have a batch size of 1."""
    dice_scores = []
    for batch in test_loader:
        inputs, labels = batch["image"][0].to(device), batch["label"][0].to(device, dtype=torch.uint8)
        patient = batch["patient"][0]

        inputs = convert_to_normalized_colour(inputs)
        predictor.set_image(inputs)

        labels = labels.cpu().numpy()

        bboxes, masks = [], []
        points, point_labels = (
            get_sam_points(labels, pos_sample_points, neg_sample_points) if use_points else (None, None)
        )

        for i, label in enumerate(labels):
            bbox = get_numpy_bounding_box(label) if use_bboxes else None
            point_coord = points[i] if points is not None else None
            point_label = point_labels[i] if point_labels is not None else None
            mask, _, _ = predictor.predict(
                box=bbox, point_coords=point_coord, point_labels=point_label, multimask_output=False
            )
            masks.append(mask)
            bboxes.append(bbox)

        save_figure(
            patient_name=patient,
            inputs=inputs,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
            out_dir=out_dir,
            points=points,
            point_labels=point_labels,
            save_to_wandb=True,
        )
        dice_scores.append(calculate_dice_for_classes(masks, labels))

    return torch.tensor(dice_scores)


def run_batch_inference(
    test_loader: DataLoader,
    sam: Sam,
    device: str | torch.device,
    out_dir: Path,
    pos_sample_points: int,
    neg_sample_points: int = 0,
    use_bboxes: bool = True,
):
    sam.eval()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dice_scores = []
    hd_scores = []
    for batch in test_loader:
        inputs, labels, patient = (
            batch["image"].to(device),
            batch["label"].to(device, dtype=torch.uint8),
            batch["patient"],
        )

        with torch.no_grad():
            masks, boxes, points, point_labels = get_batch_predictions(
                sam=sam,
                transform=resize_transform,
                inputs=inputs,
                labels=labels,
                patients=patient,
                pos_sample_points=pos_sample_points,
                neg_sample_points=neg_sample_points,
                use_bboxes=use_bboxes,
                inference=True,
            )

        hd_scores.append(calculate_hd_for_classes(masks, labels))
        dice_scores.append(calculate_dice_for_classes(masks, labels))

        # Convert to numpy before saving
        masks = masks.cpu().numpy()
        labels = labels.cpu().numpy()

        for i in range(len(masks)):
            save_figure(
                patient[i],
                inputs[i].permute(1, 2, 0).squeeze(),
                boxes[i],
                labels[i],
                masks[i],
                out_dir=out_dir,
                points=points[i],
                point_labels=point_labels[i],
                save_to_wandb=True,
            )

    return torch.tensor(dice_scores), torch.tensor(hd_scores)


def main(
    dataset: str,
    pos_sample_points: int,
    use_bboxes: bool,
    neg_sample_points: int = 0,
    num_training_cases: int | None = None,
):
    num_samples_str = f"num_samples_{pos_sample_points}"
    use_bbox_str = "with_bboxes" if use_bboxes else "no_bboxes"
    neg_samples_str = "" if neg_sample_points == 0 else f"neg_samples_{neg_sample_points}"

    print(f"Starting inference for {dataset}...")
    root_dir = Path(os.getcwd())

    data_dir, log_dir, out_dir = setup_dirs(root_dir)

    project_name = (
        f"sam_baseline_inference_{dataset}" if num_training_cases is None else f"sam_inference_finetuned_{dataset}"
    )

    wandb.init(
        project=project_name,
        name=f"{dataset}_{'_'.join(filter(None, (num_samples_str, use_bbox_str, neg_samples_str)))}",
        config={
            "dataset": dataset,
            "pos_sample_points": pos_sample_points,
            "neg_sample_points": neg_sample_points,
            "bboxes": use_bboxes,
        },
        dir=log_dir,
        mode="disabled",
        reinit=True,
    )

    out_dir = out_dir / "sam" / dataset
    if num_training_cases is not None:
        out_dir = out_dir / f"num_training_cases_{num_training_cases}"

    out_dir = out_dir / use_bbox_str / num_samples_str
    if neg_sample_points > 0:
        out_dir = out_dir / neg_samples_str

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    batch_size = 1
    spatial_dims = 2
    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if num_training_cases is not None:
        sam = setup_sam(root_dir / "models", device)
    else:
        sam = setup_sam(out_dir, device, checkpoint="best_metric_model.pth")

    dataset_helper = DatasetHelperFactory(dataset_name=dataset)
    dataset_helper = dataset_helper.dataset(
        spatial_dims=spatial_dims, data_dir=data_dir, nnunet_transforms=True, force_foreground_classes=True
    )

    test_data = dataset_helper.get_test_dataset()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    figure_dir = out_dir / "figures"
    os.makedirs(figure_dir, exist_ok=True)

    # dice_scores = run_inference(
    #     test_loader,
    #     SamPredictor(sam),
    #     device,
    #     figure_dir,
    #     use_bboxes=use_bboxes,
    #     pos_sample_points=pos_sample_points,
    #     neg_sample_points=neg_sample_points,
    # )
    dice_scores = run_batch_inference(
        test_loader,
        sam,
        device,
        figure_dir,
        use_bboxes=use_bboxes,
        pos_sample_points=pos_sample_points,
        neg_sample_points=neg_sample_points,
    )
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
        "--use_bboxes",
        "-b",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    for dataset in ["acdc", "mnms"]:
        for num_points in [2, 3, 5]:
            for use_bboxes in [True, False]:
                for neg_points in [0, 1, 2]:
                    main(dataset, num_points, use_bboxes, neg_points)
