import os

import torch
import wandb
from segment_anything.modeling import Sam
from torch.utils.data import DataLoader

from src.metrics import aggregate_validation_metrics
from src.sam.sam_utils import get_batch_predictions
from src.train import compute_val_loss_and_metrics


def train(
    sam: Sam,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_interval: int,
    epochs: int,
    device: str | torch.device,
    out_dir: str | os.PathLike,
    num_sample_points: int,
):
    sam.image_encoder.requires_grad_(False)
    sam.prompt_encoder.requires_grad_(False)

    best_metric = -1
    best_metric_epoch = -1

    epoch_loss_values = []
    metric_values: dict[
        str, list[float] | list[list[float]]
    ] = {}  # For each metric, store a list of values one for each validation epoch

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, verbose=True)

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1

            loss = get_epoch_loss(
                optimizer=optimizer,
                batch_data=batch_data,
                sam=sam,
                loss_function=loss_function,
                device=device,
                num_sample_points=num_sample_points,
            )

            if loss == -1:
                continue

            epoch_loss += loss
            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, " f"train_loss: {loss:.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        wandb.log({"epoch_loss": epoch_loss})
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            validate(
                sam=sam,
                val_loader=val_loader,
                device=device,
                loss_function=loss_function,
                metric_values=metric_values,
                num_sample_points=num_sample_points,
            )

            dice_metric = metric_values["dice_with_background"][-1]
            if dice_metric > best_metric:
                best_metric = dice_metric
                best_metric_epoch = epoch + 1
                torch.save(
                    sam.state_dict(),
                    os.path.join(out_dir, "best_metric_model.pth"),
                )
                print(f"New best metric found: {best_metric}")

            print(
                f"current epoch: {epoch + 1} current mean dice: {dice_metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

        scheduler.step(epoch_loss)

    return epoch_loss_values, metric_values


def get_epoch_loss(
    optimizer: torch.optim.Optimizer,
    batch_data: torch.tensor,
    sam: Sam,
    loss_function: torch.nn.Module,
    device: str | torch.device,
    num_sample_points: int,
    num_classes: int = 4,
):
    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

    sam.train()
    predictions, labels, _, _, _ = get_batch_predictions(
        sam=sam, inputs=inputs, labels=labels, num_points=num_sample_points, num_classes=num_classes
    )
    if predictions == [] and labels == []:
        # Empty batch due to all labels being incomplete
        return -1

    optimizer.zero_grad()
    loss = loss_function(predictions, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(
    sam: Sam,
    val_loader: DataLoader,
    device: str | torch.device,
    loss_function: torch.nn.Module,
    metric_values,
    num_sample_points: int,
):
    sam.eval()
    with torch.no_grad():
        validation_loss = 0
        step = 0
        for val_data in val_loader:
            step += 1

            validation_loss += get_validation_loss(
                val_data=val_data,
                sam=sam,
                loss_function=loss_function,
                device=device,
                num_sample_points=num_sample_points,
            )

        validation_loss /= step
        wandb.log({"validation_loss": validation_loss})
        print(f"validation_loss: {validation_loss:.4f}")

        aggregate_validation_metrics(metric_values=metric_values)


def get_validation_loss(
    val_data: torch.tensor, sam: Sam, loss_function: torch.nn.Module, device: str | torch.device, num_sample_points: int
):
    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    val_outputs, val_labels, _, _, _ = get_batch_predictions(
        sam=sam, inputs=val_inputs, labels=val_labels, num_points=num_sample_points
    )
    val_loss = compute_val_loss_and_metrics(
        inputs=val_inputs, outputs=val_outputs, labels=val_labels, loss_function=loss_function
    )
    return val_loss.item()
