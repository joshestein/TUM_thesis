import os.path
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from scipy import ndimage
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


def get_bounding_box(ground_truth_map: torch.Tensor, margin: int = 10) -> list[int]:
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
    ground_truth: np.ndarray, num_classes: int, num_pos_points: int, num_neg_points: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    # Sample n points from each class of the ground truth mask.
    # Samples are chosen by:
    # 1. Finding the centre of mass of the class.
    # 2. If the centre of mass is part of the class, it is used as the initial sampling point.
    # 3. All remaining points are sampled from a uniform distribution across a flattened vector of the class.

    # If negative points are given, we re-use the positively sampled points as negative points for other classes.

    points_per_class = [sample_points(ground_truth[class_index], num_pos_points) for class_index in range(num_classes)]
    labels_per_class = np.concatenate(
        (
            np.ones((num_classes, num_pos_points), dtype=int),
            # -1 since we sample negatively from the N-1 classes
            np.zeros((num_classes, num_neg_points * (num_classes - 1)), dtype=int),
        ),
        axis=1,
    )

    if num_neg_points > 0:
        for i in range(num_classes):
            # Instead of sampling new negative points, we use the previously sampled positive points as negative points
            # for other classes.
            random_indices = np.random.choice(num_pos_points, num_neg_points, replace=True)
            sample = points_per_class[i][random_indices]
            for j in range(num_classes):
                if i != j:
                    points_per_class[j] = np.append(points_per_class[j], sample, axis=0)

    return np.array(points_per_class), labels_per_class


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


def show_points(coords, labels, ax, marker_size=375):
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


def show_box(box, ax):
    if box is None or any(v is None for v in box):
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
    inputs: torch.Tensor,
    labels: torch.Tensor,
    patients: torch.Tensor,
    pos_sample_points: int,
    neg_sample_points: int = 0,
    use_bboxes: bool = True,
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

    # Swap width and height
    # B x C x W x H -> B x C x H x W
    labels = torch.moveaxis(labels, (2, 3), (3, 2))

    for index, image in enumerate(inputs):
        ground_truth = labels[index]
        prepared_image = prepare_image(image, transform, sam.device)

        points, point_labels = get_sam_points(
            ground_truth.cpu().numpy(), num_classes, pos_sample_points, neg_sample_points
        )

        # Get bounding box and points for each class of one-hot encoded mask
        for i, label in enumerate(ground_truth):
            if use_bboxes is not None:
                bbox = get_bounding_box(label)
                bbox = transform.apply_boxes_torch(torch.as_tensor(bbox, device=sam.device), image.shape[1:])
            else:
                bbox = None

            point = transform.apply_coords_torch(torch.as_tensor(points[i], device=sam.device), image.shape[1:])
            point_label = torch.as_tensor(point_labels[i], device=sam.device)
            point, point_label = point.unsqueeze(0), point_label.unsqueeze(0)

            batched_input.append(
                {
                    "image": prepared_image if i == 0 else None,
                    "boxes": bbox,
                    "point_coords": point,
                    "point_labels": point_label,
                    "original_size": image.shape[1:],
                    "patient": patients[index],
                }
            )

    if not batched_input:
        return [], [], [], [], [], []

    # The gradients are disabled in Sam with the decorator @torch.no_grad.
    # We re-write the forward pass here to enable gradients.
    batched_output = forward(sam, batched_input, num_classes, multimask_output=False)

    bboxes, points, point_labels, transformed_images = collate_batch_inputs(batched_input, num_classes)
    masks = [batched_output[i]["masks"] for i in range(len(batched_output))]

    return masks, labels, bboxes, points, point_labels, transformed_images


def collate_batch_inputs(batched_input, num_classes: int):
    bboxes, points, point_labels = [], [], []
    for i in range(0, len(batched_input), num_classes):
        if batched_input[i]["boxes"] is not None:
            collated_boxes = torch.stack(
                [batched_input[i + class_index]["boxes"][0] for class_index in range(num_classes)]
            )
        else:
            collated_boxes = [None] * num_classes

        if batched_input[i]["point_coords"] is not None:
            collated_points = torch.stack(
                [batched_input[i + class_index]["point_coords"][0] for class_index in range(num_classes)]
            )
        else:
            collated_points = [None] * num_classes

        if batched_input[i]["point_labels"] is not None:
            collated_point_labels = torch.stack(
                [batched_input[i + class_index]["point_labels"][0] for class_index in range(num_classes)]
            )
        else:
            collated_point_labels = [None] * num_classes

        bboxes.append(collated_boxes)
        points.append(collated_points)
        point_labels.append(collated_point_labels)
        transformed_images.append(batched_input[i]["image"].permute(1, 2, 0))  # Move channels to last dimension

    transformed_images = torch.stack(transformed_images)

    return bboxes, points, point_labels, transformed_images


def save_figure(
    patient_name: str,
    inputs,
    bboxes,
    labels,
    masks,
    out_dir: Path,
    num_classes: int = 4,
    points: list | np.ndarray | None = None,
    point_labels: list | np.ndarray | None = None,
    save_to_wandb: bool = False,
):
    plt.ioff()
    fig = plt.figure(figsize=(16, 16))
    plt.tight_layout()

    for class_index in range(num_classes):
        # Original input
        plt.subplot(num_classes, 3, class_index * 3 + 1)
        plt.imshow(inputs, cmap="gray")
        show_box(bboxes[class_index], plt.gca())

        if points and point_labels:
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


def forward(
    sam: Sam,
    batched_input: list[dict[str, any]],
    num_classes: int,
    multimask_output=False,
):
    image_embeddings = get_and_save_embeddings(sam, batched_input, num_classes)

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
            image_embeddings=curr_embedding,
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


def get_and_save_embeddings(
    sam: Sam,
    batched_input: list[dict[str, any]],
    num_classes: int,
    embeddings_dir: Path = Path(os.getcwd()) / "data" / "embeddings",
):
    image_embeddings = []
    for i in range(0, len(batched_input), num_classes):
        patient = batched_input[i]["patient"]
        if not os.path.exists(embeddings_dir / f"{patient}.pt"):
            image = sam.preprocess(batched_input[i]["image"])
            with torch.no_grad():
                embedding = sam.image_encoder(image.unsqueeze(0))
            torch.save(embedding, embeddings_dir / f"{patient}.pt")
        else:
            embedding = torch.load(embeddings_dir / f"{patient}.pt")

        # Expand the embedding to match the number of classes. Each embedding is re-used for each prediction class.
        image_embeddings.append(embedding.expand(num_classes, -1, -1, -1))

    return image_embeddings
