import os
from typing import Optional

import torch
import wandb
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from torch.utils.data import DataLoader

from src.metrics import METRICS
from src.models.early_stopper import EarlyStopper


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_interval: int,
    epochs: int,
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)

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
                model=model,
                loss_function=loss_function,
                device=device,
            )

            epoch_loss += loss
            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, " f"train_loss: {loss:.4f}")

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
                metric_values=metric_values,
            )

            dice_metric = metric_values["dice_with_background"][-1]
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

        scheduler.step(epoch_loss)

        if early_stopper and early_stopper.stop:
            print("Early stop")
            break

    return epoch_loss_values, metric_values


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str | torch.device,
    loss_function: torch.nn.Module,
    metric_values,
):
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        step = 0
        for val_data in val_loader:
            step += 1

            validation_loss += get_validation_loss(
                val_data=val_data, model=model, loss_function=loss_function, device=device
            )

        validation_loss /= step
        wandb.log({"validation_loss": validation_loss})
        print(f"validation_loss: {validation_loss:.4f}")

        for name, metric in METRICS.items():
            # aggregate and reset metrics
            metric_value = metric.aggregate()
            metric_value = metric_value.item() if len(metric_value) == 1 else metric_value.tolist()
            metric.reset()

            if name not in metric_values:
                metric_values[name] = []

            metric_values[name].append(metric_value)
            wandb.log({f"validation_{name}": metric_value})


def get_epoch_loss(
    optimizer: torch.optim.Optimizer,
    batch_data: torch.tensor,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    device: str | torch.device,
):
    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)

    if len(inputs.shape) == 5:
        # In our transforms, we use `Transpose` to rearrange into B, C, D, H, W
        # This is because 3D layers in Pytorch expect D before H, W
        # However, for Monai metrics and loss, we need to rearrange to B, C, H, W, D (put depth at the last dimension).
        # We permute after passing through the model.
        outputs = outputs.permute(0, 1, 3, 4, 2)
        labels = labels.permute(0, 1, 3, 4, 2)

    loss = loss_function(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def get_validation_loss(
    val_data: torch.tensor,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    device: str | torch.device,
):
    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    val_outputs = model(val_inputs)

    if len(val_inputs.shape) == 5:
        # Permute after passing through the model.
        val_outputs = val_outputs.permute(0, 1, 3, 4, 2)
        val_labels = val_labels.permute(0, 1, 3, 4, 2)

    val_loss = compute_val_loss_and_metrics(outputs=val_outputs, labels=val_labels, loss_function=loss_function)
    return val_loss.item()


def compute_val_loss_and_metrics(outputs, labels, loss_function):
    post_pred = Compose([AsDiscrete(to_onehot=4, argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=4)])

    val_loss = loss_function(outputs, labels)
    val_outputs = [post_pred(i) for i in decollate_batch(outputs)]
    val_labels = [post_label(i) for i in decollate_batch(labels)]

    for metric in METRICS.values():
        metric(y_pred=val_outputs, y=val_labels)

    return val_loss
