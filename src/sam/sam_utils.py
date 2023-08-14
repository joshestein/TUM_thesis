from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold


def get_bounding_box(ground_truth_map: torch.tensor):
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


def prepare_image(image, transform, device):
    image = cv2.cvtColor(image.permute(2, 1, 0).cpu().numpy(), cv2.COLOR_GRAY2RGB)
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def get_predictions(sam: Sam, inputs: torch.tensor, labels: torch.tensor, num_classes=4, transform=None):
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
            label = torch.as_tensor(ground_truth == class_index, dtype=torch.uint8, device=sam.device)
            bbox = None if torch.count_nonzero(label) == 0 else torch.tensor(get_bounding_box(label), device=sam.device)
            batched_input.append(
                {
                    "image": prepared_image,
                    "boxes": transform.apply_boxes_torch(bbox, image.shape[1:]) if bbox is not None else None,
                    "original_size": image.shape[1:],
                    "gt": label,
                }
            )

    # TODO: the gradients are disabled in Sam with the decorator @torch.no_grad
    # batched_output = sam(batched_input, multimask_output=False)
    batched_output = forward(sam, batched_input, multimask_output=False)

    masks, ground_truths, bboxes, transformed_images = [], [], [], []
    for i in range(0, len(batched_output), num_classes):
        collated_masks = [batched_output[i + class_index]["masks"].squeeze() for class_index in range(num_classes)]
        collated_gts = [batched_input[i + class_index]["gt"] for class_index in range(num_classes)]
        collated_boxes = [
            batched_input[i + class_index]["boxes"][0] if batched_input[i + class_index]["boxes"] is not None else None
            for class_index in range(num_classes)
        ]
        masks.append(torch.stack(collated_masks))
        ground_truths.append(torch.stack(collated_gts))
        bboxes.append(collated_boxes)  # Don't stack, otherwise NoneType errors
        transformed_images.append(batched_input[i]["image"].permute(1, 2, 0))  # Move channels to last dimension

    masks = torch.stack(masks)
    ground_truths = torch.stack(ground_truths)
    transformed_images = torch.stack(transformed_images)
    return masks, ground_truths, bboxes, transformed_images


def save_figure(index: int, inputs, bboxes, labels, masks, out_dir: Path, num_classes=4):
    plt.ioff()
    fig = plt.figure(figsize=(8, 8))
    for class_index in range(num_classes):
        if class_index == 0:
            continue

        # Original input
        plt.subplot(num_classes, 3, (class_index - 1) * 3 + 1)
        plt.imshow(inputs, cmap="gray")
        box = bboxes[class_index]
        if box is not None:
            show_box(box, plt.gca())
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
