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
from src.metrics import compute_surface_metrics
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
    dice_scores, hd_scores = [], []
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
    dice_scores, hd_scores, mad_scores = [], [], []
    for batch in test_loader:
        inputs, labels, patient, spacing = (
            batch["image"].to(device),
            batch["label"].to(device, dtype=torch.uint8),
            batch["patient"],
            batch["spacing"],
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

        dice_scores.append(calculate_dice_for_classes(masks, labels))

        # Convert to numpy before saving
        masks = masks.cpu().numpy()
        labels = labels.cpu().numpy()

        surface_metrics = compute_surface_metrics(masks, labels.astype(bool), spacing_mm=[s.item() for s in spacing])
        hd_scores.append(torch.as_tensor(surface_metrics["hausdorff"]))
        mad_scores.append(torch.as_tensor(surface_metrics["mean_absolute_difference"]))

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

    return (
        torch.mean(torch.stack(dice_scores), dim=0),
        torch.mean(torch.stack(hd_scores), dim=0),
        torch.mean(torch.stack(mad_scores), dim=0),
    )


def main(
    dataset: str,
    pos_sample_points: int,
    use_bboxes: bool,
    neg_sample_points: int = 0,
    num_training_cases: int | None = None,
):
    num_training_cases_str = "" if num_training_cases is None else f"num_training_cases_{num_training_cases}"
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
        name=f"{dataset}_{'_'.join(filter(None, (num_training_cases_str, num_samples_str, use_bbox_str, neg_samples_str)))}",
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
        out_dir = out_dir / f"num_training_cases_{num_training_cases}" / num_samples_str
    else:
        out_dir = out_dir / use_bbox_str / num_samples_str

    if neg_sample_points > 0:
        out_dir = out_dir / neg_samples_str

    with open(root_dir / "config.toml", "rb") as file:
        config = tomllib.load(file)

    batch_size = 1
    spatial_dims = 2
    set_determinism(seed=config["hyperparameters"]["seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if num_training_cases is None:
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
    dice_scores, hd_scores, mad_scores = [], [], []

    # Average over 5 runs
    for i in range(5):
        dice, hd, mad = run_batch_inference(
            test_loader,
            sam,
            device,
            figure_dir,
            use_bboxes=use_bboxes,
            pos_sample_points=pos_sample_points,
            neg_sample_points=neg_sample_points,
        )
        dice_scores.append(dice)
        hd_scores.append(hd)
        mad_scores.append(mad)

    dice_scores = torch.stack(dice_scores)
    hd_scores = torch.stack(hd_scores)
    mad_scores = torch.stack(mad_scores)

    dice_per_class, dice_std_per_class, dice_mean, dice_std = get_mean_and_std(dice_scores, "dice")
    hd_per_class, hd_std_per_class, hd_mean, hd_std = get_mean_and_std(hd_scores, "hd")
    mad_per_class, mad_std_per_class, mad_mean, mad_std = get_mean_and_std(mad_scores, "mad")

    results = {
        "dice_per_class": dice_per_class,
        "mean_dice": dice_mean,
        "mean_dice_std": dice_std,
        "hd_per_class": hd_per_class,
        "mean_hd": hd_mean,
        "mean_hd_std": hd_std,
        "mad_per_class": mad_per_class,
        "mean_mad": mad_mean,
        "mean_mad_std": mad_std,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


def get_mean_and_std(tensor: torch.Tensor, label: str):
    score_per_class = torch.mean(tensor, dim=0)
    std_per_class = torch.std(tensor, dim=0)
    mean_score = torch.mean(score_per_class).item()
    mean_std = torch.mean(std_per_class).item()
    print(f"{label} per class: {score_per_class}")
    print(f"{label} std per class: {score_per_class}")
    print(f"{label} mean: {mean_score}")
    print(f"{label} std: {mean_std}")

    score_per_class = score_per_class.tolist()
    std_per_class = std_per_class.tolist()

    wandb.log({f"{label}_per_class": score_per_class})
    wandb.log({f"{label}_std_per_class": std_per_class})
    wandb.log({f"mean_{label}": mean_score})
    wandb.log({f"mean_{label}_std": mean_std})

    return score_per_class, std_per_class, mean_score, mean_std


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

    dataset_case_dict = {"acdc": [8, 24, 32, 48, 80, 160], "mnms": [8, 24, 32, 48, 80, 160, 192, 240]}

    for dataset in ["acdc", "mnms"]:
        # Baseline inference
        for num_points in [2, 3, 5]:
            for use_bboxes in [True, False]:
                for neg_points in [0, 1, 2]:
                    main(dataset, num_points, use_bboxes, neg_points)

        # Fine-tuned inference
        for num_cases in dataset_case_dict[dataset]:
            # We only fine-tune with 2 positive sample points, 1 neg sample point, with bboxes
            main(dataset, 2, True, 1, num_cases)
