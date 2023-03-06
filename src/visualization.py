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
            for i in range(len(metric_values[0])):
                plt.plot(
                    [val_interval * (i + 1) for i in range(len(metric_values))],
                    [m[i] for m in metric_values],
                    label=f"{metric_name}_{i}",
                )
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
    image_key: str,
    label_key: str,
    slice_no: int = 0,
):
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_outputs = model(val_data["end_diastole"].to(device))
            plt.figure("Check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"Image {i}")
            plt.imshow(val_data[image_key][0, 0, :, :, slice_no], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"Label {i}")
            plt.imshow(val_data[label_key][0, 0, :, :, slice_no])
            plt.subplot(1, 3, 3)
            plt.title(f"Output {i}")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_no])
            plt.show()
            if i == 2:
                break
