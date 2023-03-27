import os
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

            # In our transforms, we use `Transpose` to rearrange into B, C, D, H, W
            # This is because most 3D layers in Pytorch expect D before H, W
            # However, for Monai metrics and dice loss, we need to rearrange to B, C, H, W, D
            for key, image in batch_data.items():
                batch_data[key] = image.permute(0, 1, 3, 4, 2)

            end_diastole, end_systole, end_diastole_labels, end_systole_labels = (
                batch_data["end_diastole"].to(device),
                batch_data["end_systole"].to(device),
                batch_data["end_diastole_label"].to(device),
                batch_data["end_systole_label"].to(device),
            )

            inputs = torch.vstack((end_diastole, end_systole))
            labels = torch.vstack((end_diastole_labels, end_systole_labels))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        wandb.log({"epoch_loss": epoch_loss})
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            validate(
                model=model,
                val_loader=val_loader,
                device=device,
                loss_function=loss_function,
                metrics=metrics,
                metric_values=metric_values,
            )

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
    loss_function: torch.nn.Module,
    metrics: dict[str, CumulativeIterationMetric],
    metric_values,
):
    post_pred = Compose([AsDiscrete(to_onehot=4, argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=4)])

    model.eval()
    with torch.no_grad():
        validation_loss = 0
        step = 0
        for val_data in val_loader:
            step += 1
            for key, image in val_data.items():
                val_data[key] = image.permute(0, 1, 3, 4, 2)

            end_diastole, end_systole, end_diastole_labels, end_systole_labels = (
                val_data["end_diastole"].to(device),
                val_data["end_systole"].to(device),
                val_data["end_diastole_label"].to(device),
                val_data["end_systole_label"].to(device),
            )

            val_inputs = torch.vstack((end_diastole, end_systole))
            val_labels = torch.vstack((end_diastole_labels, end_systole_labels))
            val_outputs = model(val_inputs)

            loss = loss_function(val_outputs, val_labels)
            validation_loss += loss.item()

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            for metric in metrics.values():
                # compute metrics for current iteration
                metric(y_pred=val_outputs, y=val_labels)

        validation_loss /= step
        wandb.log({"validation_loss": validation_loss})
        print(f"validation_loss: {validation_loss:.4f}")

        for name, metric in metrics.items():
            # aggregate and reset metrics
            metric_value = metric.aggregate()
            metric_value = metric_value.item() if len(metric_value) == 1 else metric_value.tolist()
            metric.reset()

            if name not in metric_values:
                metric_values[name] = []

            metric_values[name].append(metric_value)
            wandb.log({f"validation_{name}": metric_value})
