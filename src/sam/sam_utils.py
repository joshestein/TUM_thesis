from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from scipy import ndimage
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


def get_bounding_box(ground_truth_map: torch.tensor, margin: int = 10) -> list[torch.tensor]:
    # get bounding box from mask
    y_indices, x_indices = torch.where(ground_truth_map > 0)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    # add perturbation to bounding box coordinates
    height, width = ground_truth_map.shape
    x_min = max(0, x_min - margin)
    x_max = min(width, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(height, y_max + margin)
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


def get_numpy_bounding_box(ground_truth_map: np.ndarray, margin: int = 10) -> np.ndarray:
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    height, width = ground_truth_map.shape
    x_min = max(0, x_min - margin)
    x_max = min(width, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(height, y_max + margin)
    bbox = [x_min, y_min, x_max, y_max]

    return np.array(bbox)


def get_sam_points(
    ground_truth: np.ndarray, num_classes: int, num_pos_points: int, num_neg_points: int = 1
) -> (np.array, np.array):
    # Sample n points from each class of the ground truth mask.
    # Samples are chosen by:
    # 1. Finding the centre of mass of the class.
    # 2. If the centre of mass is part of the class, it is used as the initial sampling point.
    # 3. All remaining points are sampled from a uniform distribution across a flattened vector of the class.

    # Negative points are sampled from the ventricles when sampling from the myocardium.
    points_per_class, labels_per_class = [], []

    for class_index in range(num_classes):
        sampled_points = sample_points((ground_truth == class_index).astype(int), num_pos_points)
        ones = np.ones(num_pos_points)

        # For the myocardium, we sample negative points from both ventricles as well
        if class_index == 2:
            lv_points = sample_points((ground_truth == 1).astype(int), num_neg_points)
            rv_points = sample_points((ground_truth == 3).astype(int), num_neg_points)

            sampled_points = np.concatenate((sampled_points, lv_points, rv_points))
            ones = np.concatenate((ones, np.zeros(num_neg_points), np.zeros(num_neg_points)))

        points_per_class.append(sampled_points)
        labels_per_class.append(ones)

    # Use dtype=object since the arrays will be jagged, due to class 2 sampling negative points
    return np.array(points_per_class, dtype=object), np.array(labels_per_class, dtype=object)


def sample_points(mask: np.ndarray, num_points: int) -> np.ndarray:
    mask = mask.T  # Since we swap the W and H during the dataloading, we reverse that swap here.
    center_of_mass = ndimage.measurements.center_of_mass(mask)
    center_of_mass = [int(i) for i in center_of_mass]
    if mask[*center_of_mass] == 1:
        points = [center_of_mass]
        num_points -= 1
    else:
        points = []

    indices = np.argwhere(mask == 1)
    sampled_points = indices[np.random.choice(len(indices), num_points, replace=False)]
    points.extend(sampled_points)

    return np.array(points)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords: np.ndarray | torch.tensor, labels: np.ndarray | torch.tensor, ax, marker_size=375):
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box: np.ndarray | torch.tensor, ax):
    if any(v is None for v in box):
        return

    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))  # %%


def convert_to_normalized_colour(image):
    """Convert to colour, scale to 0-255, convert to uint8"""
    image = cv2.cvtColor(image.permute(2, 1, 0).cpu().numpy(), cv2.COLOR_GRAY2RGB)
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
    return image


def prepare_image(image, transform, device):
    image = convert_to_normalized_colour(image)
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def get_batch_predictions(
    sam: Sam,
    inputs: torch.tensor,
    labels: torch.tensor,
    patients: torch.tensor,
    num_points: int,
    num_classes=4,
    transform=None,
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
        onehot_labels = [
            torch.as_tensor((ground_truth == class_index), dtype=torch.int, device=sam.device)
            for class_index in range(num_classes)
        ]

        if any(torch.count_nonzero(label) == 0 for label in onehot_labels):
            print(f"Skipping patient {patients[index]} as it contains empty labels.")
            continue

        points, point_labels = get_sam_points(ground_truth.cpu().numpy(), num_classes, num_points)

        # Get bounding box and points for each class of one-hot encoded mask
        for i, label in enumerate(onehot_labels):
            bbox = get_bounding_box(label)
            point, point_label = None, None

            if bbox is not None:
                bbox = transform.apply_boxes_torch(torch.as_tensor(bbox, device=sam.device), image.shape[1:])
            if points[i] is not None:
                point = transform.apply_coords_torch(torch.as_tensor(points[i], device=sam.device), image.shape[1:])
                point_label = torch.as_tensor(point_labels[i], device=sam.device)
                point, point_label = point.unsqueeze(0), point_label.unsqueeze(0)

            batched_input.append(
                {
                    "image": prepared_image,
                    "boxes": bbox,
                    "point_coords": point,
                    "point_labels": point_label,
                    "original_size": image.shape[1:],
                    "gt": label,
                }
            )

    if not batched_input:
        return [], [], [], [], [], []

    # TODO: the gradients are disabled in Sam with the decorator @torch.no_grad
    # batched_output = sam(batched_input, multimask_output=False)
    batched_output = forward(sam, batched_input, multimask_output=False)

    masks = collate_mask_outputs(batched_output, num_classes)
    ground_truths, bboxes, points, point_labels, transformed_images = collate_batch_inputs(batched_input, num_classes)

    return masks, ground_truths, bboxes, points, point_labels, transformed_images


def collate_mask_outputs(batched_output, num_classes: int):
    masks = []
    for i in range(0, len(batched_output), num_classes):
        collated_masks = [batched_output[i + class_index]["masks"].squeeze() for class_index in range(num_classes)]
        masks.append(torch.stack(collated_masks))

    return torch.stack(masks)


def collate_batch_inputs(batched_input, num_classes: int):
    ground_truths, bboxes, points, point_labels, transformed_images = [], [], [], [], []
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
        collated_point_labels = [
            batched_input[i + class_index]["point_labels"][0]
            if batched_input[i + class_index]["point_labels"] is not None
            else None
            for class_index in range(num_classes)
        ]
        ground_truths.append(torch.stack(collated_gts))
        bboxes.append(collated_boxes)  # Don't stack, otherwise NoneType errors
        points.append(collated_points)
        point_labels.append(collated_point_labels)
        transformed_images.append(batched_input[i]["image"].permute(1, 2, 0))  # Move channels to last dimension

    ground_truths = torch.stack(ground_truths)
    transformed_images = torch.stack(transformed_images)

    return ground_truths, bboxes, points, point_labels, transformed_images


def save_figure(
    patient_name: str,
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
        # Original input
        plt.subplot(num_classes, 3, class_index * 3 + 1)
        plt.imshow(inputs)
        show_box(bboxes[class_index], plt.gca())
        show_points(points[class_index], point_labels[class_index], plt.gca(), marker_size=100)

        plt.axis("off")

        # Ground truth
        plt.subplot(num_classes, 3, class_index * 3 + 2)
        plt.imshow(labels[class_index])
        plt.axis("off")

        # Prediction
        plt.subplot(num_classes, 3, class_index * 3 + 3)
        show_mask(masks[class_index], plt.gca())
        plt.axis("off")

    plt.savefig(out_dir / f"{patient_name}.png")
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

    labels = np.argmax(labels, axis=0)
    int_masks = [(mask.astype(int) * i).squeeze() for i, mask in enumerate(masks)]
    mask_data = np.argmax(int_masks, axis=0)

    wb_image = wandb.Image(
        inputs,
        masks={
            "predictions": {"mask_data": mask_data, "class_labels": class_labels, "box_data": box_data},
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
        # masks = threshold(masks, sam.mask_threshold, 0)
        # masks = masks > sam.mask_threshold
        # Don't threshold, it's not differentiable.
        # It makes sense for prediction, but not for training.
        # TODO: should this be softmax?
        masks = torch.sigmoid(masks)
        outputs.append(
            {
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            }
        )
    return outputs
