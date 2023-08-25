from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold


def get_bounding_box(ground_truth_map: torch.tensor) -> list[torch.tensor]:
    # get bounding box from mask
    y_indices, x_indices = torch.where(ground_truth_map > 0)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    # add perturbation to bounding box coordinates
    height, width = ground_truth_map.shape
    x_min = max(0, x_min - torch.randint(0, 20, size=()))
    x_max = min(width, x_max + torch.randint(0, 20, size=()))
    y_min = max(0, y_min - torch.randint(0, 20, size=()))
    y_max = min(height, y_max + torch.randint(0, 20, size=()))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


def get_numpy_bounding_box(ground_truth_map: np.ndarray) -> np.ndarray:
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    height, width = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(width, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(height, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return np.array(bbox)


def get_points(onehot_mask: np.ndarray, num_points_to_sample: int) -> np.ndarray:
    # Sample n points from the ground truth mask.
    # Assumes the mask is one-hot encoded, and that there are at least n points in the mask.
    # Samples are chosen in a way that tries to maximize the distance between points
    nonzero_coords = np.array(np.where(onehot_mask == 1)).T

    # Randomly select the initial point
    initial_point_idx = np.random.randint(len(nonzero_coords))

    # We flip because np.where returns (y, x) coordinates, but we want (x, y)
    sampled_points = [np.flip(nonzero_coords[initial_point_idx])]
    remaining_coords = np.delete(nonzero_coords, initial_point_idx, axis=0)

    while len(sampled_points) < num_points_to_sample:
        distances = cdist(sampled_points, remaining_coords, metric="euclidean")
        min_distances = np.min(distances, axis=0)
        max_min_distance_idx = np.argmax(min_distances)

        new_point = remaining_coords[max_min_distance_idx]
        sampled_points.append(np.flip(new_point))
        remaining_coords = np.delete(remaining_coords, max_min_distance_idx, axis=0)

    return np.array(sampled_points)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))  # %%


def convert_to_normalized_grayscale(image):
    """Convert to grayscale, scale to 0-255, convert to uint8"""
    image = cv2.cvtColor(image.permute(2, 1, 0).cpu().numpy(), cv2.COLOR_GRAY2RGB)
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
    return image


def prepare_image(image, transform, device):
    image = convert_to_normalized_grayscale(image)
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def get_sam_bbox_and_points(label, num_sample_points):
    if np.count_nonzero(label) == 0:
        bbox = None
        point_coords = None
        point_label = None
    else:
        bbox = get_numpy_bounding_box(label)
        point_coords = get_points(label, num_sample_points)
        point_label = np.ones(num_sample_points)

    return bbox, point_coords, point_label


def get_predictions(
    sam: Sam, inputs: torch.tensor, labels: torch.tensor, num_points: int, num_classes=4, transform=None
):
    """Given inputs and labels, runs inference on all examples in the batch.
    For each example, returns the predicted masks, ground truth masks, bounding boxes, and transformed images.

    Note that for each input image, there are num_classes predictions, one for each class.
    """
    batched_input = []
    if transform is None:
        transform = ResizeLongestSide(sam.image_encoder.img_size)

    for index, image in enumerate(inputs):
        ground_truth = labels[index][0].permute(1, 0)  # Swap W, H
        prepared_image = prepare_image(image, transform, sam.device)

        # Get bounding box for each class of one-hot encoded mask
        for class_index in range(num_classes):
            label = np.array((ground_truth == class_index).astype(int))
            bbox, point, point_labels = get_sam_bbox_and_points(label, num_points)

            if bbox is not None:
                bbox = transform.apply_boxes_torch(torch.as_tensor(bbox, device=sam.device), image.shape[1:])
            if point is not None:
                point = transform.apply_coords_torch(torch.as_tensor(point, device=sam.device), image.shape[1:])
                point_labels = torch.as_tensor(point_labels, device=sam.device)

            batched_input.append(
                {
                    "image": prepared_image,
                    "boxes": bbox,
                    "point_coords": point,
                    "point_labels": point_labels,
                    "original_size": image.shape[1:],
                    "gt": label,
                }
            )

    # TODO: the gradients are disabled in Sam with the decorator @torch.no_grad
    # batched_output = sam(batched_input, multimask_output=False)
    batched_output = forward(sam, batched_input, multimask_output=False)

    masks = collate_mask_outputs(batched_output, num_classes)
    ground_truths, bboxes, points, transformed_images = collate_batch_inputs(batched_input, num_classes)

    return masks, ground_truths, bboxes, points, transformed_images


def collate_mask_outputs(batched_output, num_classes: int):
    masks = []
    for i in range(0, len(batched_output), num_classes):
        collated_masks = [batched_output[i + class_index]["masks"].squeeze() for class_index in range(num_classes)]
        masks.append(torch.stack(collated_masks))

    return torch.stack(masks)


def collate_batch_inputs(batched_input, num_classes: int):
    ground_truths, bboxes, points, transformed_images = [], [], [], []
    for i in range(0, len(batched_input), num_classes):
        collated_gts = [batched_input[i + class_index]["gt"] for class_index in range(num_classes)]
        collated_boxes = [
            batched_input[i + class_index]["boxes"][0] if batched_input[i + class_index]["boxes"] is not None else None
            for class_index in range(num_classes)
        ]
        collated_points = [
            batched_input[i + class_index]["point_coords"][0]
            if batched_input[i + class_index]["point_coords"] is not None
            else None
            for class_index in range(num_classes)
        ]
        ground_truths.append(np.stack(collated_gts))
        bboxes.append(collated_boxes)  # Don't stack, otherwise NoneType errors
        points.append(collated_points)
        transformed_images.append(batched_input[i]["image"].permute(1, 2, 0))  # Move channels to last dimension

    ground_truths = np.stack(ground_truths)
    transformed_images = torch.stack(transformed_images)

    return ground_truths, bboxes, points, transformed_images


def save_figure(
    index: int,
    inputs,
    bboxes,
    labels,
    masks,
    out_dir: Path,
    num_classes=4,
    points: list = None,
    point_labels: list | np.ndarray = None,
    save_to_wandb: bool = False,
):
    plt.ioff()
    fig = plt.figure(figsize=(8, 8))
    plt.tight_layout()

    for class_index in range(num_classes):
        if class_index == 0:
            continue

        # Original input
        plt.subplot(num_classes, 3, (class_index - 1) * 3 + 1)
        plt.imshow(inputs, cmap="gray")
        box = bboxes[class_index]
        if box is not None:
            show_box(box, plt.gca())

        if points is not None and points[class_index] is not None:
            show_points(points[class_index], point_labels[class_index], plt.gca(), marker_size=100)

        plt.axis("off")

        # Ground truth
        plt.subplot(num_classes, 3, (class_index - 1) * 3 + 2)
        plt.imshow(labels[class_index])
        plt.axis("off")

        # Prediction
        plt.subplot(num_classes, 3, (class_index - 1) * 3 + 3)
        show_mask(masks[class_index], plt.gca())
        plt.axis("off")

    plt.savefig(out_dir / f"{index:03d}.png")
    plt.close(fig)

    if save_to_wandb:
        save_wandb_image(inputs, bboxes, labels, masks)


def save_wandb_image(inputs, bboxes, labels, masks):
    class_labels = {0: "background", 1: "LV", 2: "MYO", 3: "RV"}
    box_data = [
        {"position": {"minX": box[0], "minY": box[1], "maxX": box[2], "maxY": box[3]}}
        for box in bboxes
        if box is not None
    ]

    masks = np.argmax(masks, axis=0).squeeze()
    labels = np.argmax(labels, axis=0)

    wb_image = wandb.Image(
        inputs,
        masks={
            "predictions": {"mask_data": masks, "class_labels": class_labels, "box_data": box_data},
            "ground_truth": {"mask_data": labels, "class_labels": class_labels},
        },
    )
    wandb.log({"sam_inference": wb_image})


def calculate_dice_for_classes(masks, labels, ignore_background=True, num_classes=4, eps=1e-6):
    dice_scores = []
    for class_index in range(num_classes):
        if ignore_background and class_index == 0:
            continue

        dice = _calculate_dice(masks[class_index], labels[class_index], eps=eps)
        dice_scores.append(dice)

    return dice_scores


def _calculate_dice(prediction, ground_truth, eps=1e-6):
    tp = prediction * ground_truth
    fp = prediction * (1 - ground_truth)
    fn = (1 - prediction) * ground_truth
    # tn = (1 - mask) * (1 - ground_truth)

    dice = (2 * tp.sum() + eps) / (2 * tp.sum() + fp.sum() + fn.sum() + eps)
    return dice


def forward(sam: Sam, batched_input: list[dict[str, any]], multimask_output=False):
    input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)

    with torch.no_grad():
        image_embeddings = sam.image_encoder(input_images)

    outputs = []
    for image_record, curr_embedding in zip(batched_input, image_embeddings):
        if "point_coords" in image_record:
            points = (image_record["point_coords"], image_record["point_labels"])
        else:
            points = None

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=curr_embedding.unsqueeze(0),
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = sam.postprocess_masks(
            low_res_masks,
            input_size=image_record["image"].shape[-2:],
            original_size=image_record["original_size"],
        )
        masks = threshold(masks, sam.mask_threshold, 0)
        # masks = masks > sam.mask_threshold
        outputs.append(
            {
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        )
    return outputs
