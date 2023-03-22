import os
from statistics import mean
from typing import Optional

import torch
import wandb
from monai.data import decollate_batch
from monai.metrics import CumulativeIterationMetric
from monai.transforms import AsDiscrete, Compose
from torch.utils.data import DataLoader

from src.models.early_stopper import EarlyStopper


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function,
    optimizer: torch.optim.Optimizer,
    val_interval: int,
    epochs: int,
    metrics: dict[str, CumulativeIterationMetric],
    device: str | torch.device,
    out_dir: str | os.PathLike,
    early_stopper: Optional[EarlyStopper] = None,
):
    best_metric = -1
    best_metric_epoch = -1

    epoch_loss_values = []
    metric_values: dict[
        str, list[float] | list[list[float]]
    ] = {}  # For each metric, store a list of values one for each validation epoch

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            end_diastole, end_systole, end_diastole_labels, end_systole_labels = (
                batch_data["end_diastole"].to(device),
                batch_data["end_systole"].to(device),
                batch_data["end_diastole_label"].to(device),
                batch_data["end_systole_label"].to(device),
            )

            _, _, _, _, slices = end_diastole.shape
            slice_losses = []
            for slice_index in range(slices):
                # TODO: loop loss twice instead?
                inputs = torch.vstack((end_diastole[..., slice_index], end_systole[..., slice_index]))
                labels = torch.vstack((end_diastole_labels[..., slice_index], end_systole_labels[..., slice_index]))

                optimizer.zero_grad()
                outputs = model(inputs)
                slice_loss = loss_function(outputs, labels)
                slice_loss.backward()
                slice_losses.append(slice_loss.item())
                optimizer.step()

            loss = mean(slice_losses)
            epoch_loss += loss
            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, " f"train_loss: {loss:.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        wandb.log({"epoch_loss": epoch_loss})
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            validate(model=model, val_loader=val_loader, device=device, metrics=metrics, metric_values=metric_values)

            dice_metric = metric_values["dice"][-1]
            early_stopper.check_early_stop(dice_metric) if early_stopper else None

            if dice_metric > best_metric:
                best_metric = dice_metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(out_dir, "best_metric_model.pth"),
                )
                print(f"New best metric found: {best_metric}")

            print(
                f"current epoch: {epoch + 1} current mean dice: {dice_metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

        if early_stopper and early_stopper.stop:
            print("Early stop")
            break

    return epoch_loss_values, metric_values


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str | torch.device,
    metrics: dict[str, CumulativeIterationMetric],
    metric_values,
):
    post_pred = Compose([AsDiscrete(to_onehot=4, argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=4)])

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            end_diastole, end_systole, end_diastole_labels, end_systole_labels = (
                val_data["end_diastole"].to(device),
                val_data["end_systole"].to(device),
                val_data["end_diastole_label"].to(device),
                val_data["end_systole_label"].to(device),
            )

            _, _, _, _, slices = end_diastole.shape
            for slice_index in range(slices):
                val_inputs = torch.vstack((end_diastole[..., slice_index], end_systole[..., slice_index]))
                val_labels = torch.vstack((end_diastole_labels[..., slice_index], end_systole_labels[..., slice_index]))
                val_outputs = model(val_inputs)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                for metric in metrics.values():
                    # compute metrics for current iteration
                    metric(y_pred=val_outputs, y=val_labels)

        for name, metric in metrics.items():
            # aggregate and reset metrics
            metric_value = metric.aggregate()
            metric_value = metric_value.item() if len(metric_value) == 1 else metric_value.tolist()
            metric.reset()

            if name not in metric_values:
                metric_values[name] = []

            metric_values[name].append(metric_value)
            wandb.log({f"validation_{name}": metric_value})
