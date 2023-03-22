from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def visualize_loss_curves(
    train_loss_values, validation_metrics: dict[str, list[float] | list[list[float]]], val_interval: int, out_dir: Path
):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(train_loss_values))]
    plt.xlabel("epoch")
    plt.plot(x, train_loss_values)
    plt.subplot(1, 2, 2)
    plt.title("Val Metrics")
    for metric_name, metric_values in validation_metrics.items():
        if isinstance(metric_values[0], list):
            continue
            # for i in range(len(metric_values[0])):
            #     plt.plot(
            #         [val_interval * (i + 1) for i in range(len(metric_values))],
            #         [m[i] for m in metric_values],
            #         label=f"{metric_name}_{i}",
            #     )
        else:
            plt.plot([val_interval * (i + 1) for i in range(len(metric_values))], metric_values, label=metric_name)

    plt.xlabel("epoch")
    plt.savefig(out_dir / "loss_curves.png")
    plt.show()


def visualize_predictions(
    model: torch.nn.Module,
    model_file: str | Path,
    val_loader: DataLoader,
    device: str | torch.device,
    image_key: str = "end_diastole",
    label_key: str = "end_diastole_label",
    slice_no: int = 0,
):
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            image, label = val_data[image_key].to(device), val_data[label_key].to(device)

            if model.dimensions == 2:
                image = image[..., slice_no]
                label = label[..., slice_no]

            visualize_slice(image, label, model(image), i, slice_no)

            if i == 2:
                break


def visualize_slice(image: torch.Tensor, label: torch.Tensor, prediction: torch.Tensor, index: int, slice_no=0):
    # Get first image/label in batch, squeeze out channel dimension
    image = image[0].squeeze()
    label = label[0].squeeze()
    prediction = torch.argmax(prediction, dim=1)[0].squeeze().detach().cpu()

    # 3D input will be H x W x D, where D is number of slices - take only the `slice_no` slice
    if image.ndim == 3:
        image = image[..., slice_no]
        label = label[..., slice_no]
        prediction = prediction[..., slice_no]

    plt.figure("Check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Image {index}")
    plt.imshow(image, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title(f"Label {index}")
    plt.imshow(label)
    plt.subplot(1, 3, 3)
    plt.title(f"Prediction {index}")
    plt.imshow(prediction)
    plt.show()
