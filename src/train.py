import os
from typing import Optional

import torch
import wandb
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from torch.utils.data import DataLoader

from src.metrics import MetricHandler
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
    metric_handler: MetricHandler,
    early_stopper: Optional[EarlyStopper] = None,
):
    epoch_loss_values = []
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
                metric_handler=metric_handler,
            )

            dice_metric = metric_handler.last_value()
            early_stopper.check_early_stop(dice_metric) if early_stopper else None

            if metric_handler.check_best_metric(epoch + 1):
                torch.save(
                    model.state_dict(),
                    os.path.join(out_dir, "best_metric_model.pth"),
                )
                print(f"New best metric found: {metric_handler.best_metric}")

            print(
                f"current epoch: {epoch + 1} current mean dice: {dice_metric:.4f}"
                f"\nbest mean dice: {metric_handler.best_metric:.4f} "
                f"at epoch: {metric_handler.best_epoch}"
            )

        scheduler.step(epoch_loss)

        if early_stopper and early_stopper.stop:
            print("Early stop")
            break

    return epoch_loss_values


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str | torch.device,
    loss_function: torch.nn.Module,
    metric_handler: MetricHandler,
):
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        step = 0
        for val_data in val_loader:
            step += 1

            validation_loss += get_validation_loss(
                val_data=val_data,
                model=model,
                loss_function=loss_function,
                device=device,
                metric_handler=metric_handler,
            )

        validation_loss /= step
        wandb.log({"validation_loss": validation_loss})
        print(f"validation_loss: {validation_loss:.4f}")

        metric_handler.aggregate_and_reset_metrics()


def get_epoch_loss(
    optimizer: torch.optim.Optimizer,
    batch_data: torch.Tensor,
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    device: str | torch.device,
):
    inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)

    # if len(inputs.shape) == 5:
    #     # In our transforms, we use `Transpose` to rearrange into B, C, D, H, W
    #     # This is because 3D layers in Pytorch expect D before H, W
    #     # However, for Monai metrics and loss, we need to rearrange to B, C, H, W, D (put depth at the last dimension).
    #     # We permute after passing through the model.
    #     outputs = outputs.permute(0, 1, 3, 4, 2)
    #     labels = labels.permute(0, 1, 3, 4, 2)

    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def get_validation_loss(
    val_data: dict[str, torch.Tensor],
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    device: str | torch.device,
    metric_handler: MetricHandler,
):
    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    val_outputs = model(val_inputs)

    # if len(val_inputs.shape) == 5:
    #     # Permute after passing through the model.
    #     val_outputs = val_outputs.permute(0, 1, 3, 4, 2)
    #     val_labels = val_labels.permute(0, 1, 3, 4, 2)

    val_loss = compute_val_loss_and_metrics(
        inputs=val_inputs,
        outputs=val_outputs,
        labels=val_labels,
        loss_function=loss_function,
        metric_handler=metric_handler,
    )
    return val_loss.item()


def compute_val_loss_and_metrics(inputs, outputs, labels, loss_function, metric_handler: MetricHandler):
    post_pred = Compose([AsDiscrete(to_onehot=4, argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=4)])

    val_loss = loss_function(outputs, labels)
    val_outputs = [post_pred(i) for i in decollate_batch(outputs)]

    val_labels = decollate_batch(labels)
    if val_labels[0].shape[0] == 1:
        # Otherwise the label is already in a one-hot form
        val_labels = [post_label(i) for i in val_labels]

    class_labels = {0: "background", 1: "RV", 2: "MYO", 3: "LV"}

    wb_image = wandb.Image(
        inputs[0][0].cpu(),
        masks={
            "predictions": {
                "mask_data": torch.argmax(val_outputs[0], dim=0).cpu().numpy(),
                "class_labels": class_labels,
            },
            "ground_truth": {"mask_data": torch.argmax(val_outputs[0]).cpu().numpy(), "class_labels": class_labels},
        },
    )

    wandb.log({"predictions": wb_image})
    metric_handler.accumulate_metrics(val_outputs, val_labels)

    return val_loss
