from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def get_bounding_box(ground_truth_map):
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
    image = cv2.cvtColor(image.permute(2, 1, 0).numpy(), cv2.COLOR_GRAY2RGB)
    image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype("uint8")
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def save_single_figure(index: int, inputs, bboxes, labels, masks, out_dir: Path, num_classes=4):
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    for class_index in range(num_classes):
        if class_index == 0:
            continue

        plt.subplot(num_classes, 3, class_index * 3 + 1)
        # Original input
        plt.imshow(inputs)
        box = bboxes[class_index]
        if box is not None:
            show_box(box, plt.gca())
        plt.axis("off")

        # Ground truth
        plt.subplot(num_classes, 3, class_index * 3 + 2)
        plt.imshow(labels[class_index])
        plt.axis("off")

        # Prediction
        plt.subplot(num_classes, 3, class_index * 3 + 3)
        show_mask(masks[class_index], plt.gca())
        plt.axis("off")

    plt.savefig(out_dir / f"{index:03d}.png")
    plt.close(fig)


def save_figures(batch_index: int, batched_input, batched_output, out_dir: Path, num_classes=4):
    for i in range(0, len(batched_output), num_classes):
        # Move channels last
        inputs = batched_input[i]["image"].permute(1, 2, 0).detach().cpu()
        bboxes = [batched_input[i + class_index]["box"][0] for class_index in range(num_classes)]
        labels = [batched_input[i + class_index]["label"] for class_index in range(num_classes)]
        masks = [batched_output[i + class_index]["masks"].detach().cpu() for class_index in range(num_classes)]
        save_single_figure(batch_index * 4 + i, inputs, bboxes, labels, masks, out_dir, num_classes=num_classes)


def calculate_dice_for_classes(masks, labels, ignore_background=True, num_classes=4, eps=1e-6):
    dice_scores = []
    for class_index in range(num_classes):
        if ignore_background and class_index == 0:
            continue

        dice = _calculate_dice(masks[class_index], labels[class_index])
        dice_scores.append(dice)

    return dice_scores


def calculate_dice_from_sam_batch(batched_input, batched_output, ignore_background=True, num_classes=4):
    dice_scores = []
    for i in range(0, len(batched_output), num_classes):
        masks = [batched_output[i + class_index]["masks"].long() for class_index in range(num_classes)]
        labels = [batched_input[i + class_index]["label"] for class_index in range(num_classes)]
        dice = calculate_dice_for_classes(masks, labels, ignore_background=ignore_background, num_classes=num_classes)
        dice_scores.append(dice)

    return dice_scores


def _calculate_dice(prediction, ground_truth, eps=1e-6):
    tp = prediction * ground_truth
    fp = prediction * (1 - ground_truth)
    fn = (1 - prediction) * ground_truth
    # tn = (1 - mask) * (1 - ground_truth)

    dice = (2 * tp.sum() + eps) / (2 * tp.sum() + fp.sum() + fn.sum() + eps)
    return dice
