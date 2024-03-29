import os

import torch
import wandb
from segment_anything.modeling import Sam
from torch.utils.data import DataLoader

from src.metrics import MetricHandler
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
    pos_sample_points: int,
    metric_handler: MetricHandler,
    neg_sample_points: int = 0,
    use_bboxes=True,
):
    sam.image_encoder.requires_grad_(False)
    sam.prompt_encoder.requires_grad_(False)

    epoch_loss_values = []

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
                sam=sam,
                loss_function=loss_function,
                device=device,
                pos_sample_points=pos_sample_points,
                neg_sample_points=neg_sample_points,
                use_bboxes=use_bboxes,
            )

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
                pos_sample_points=pos_sample_points,
                neg_sample_points=neg_sample_points,
                use_bboxes=use_bboxes,
                metric_handler=metric_handler,
            )

            dice_metric = metric_handler.last_value()
            if metric_handler.check_best_metric(epoch + 1):
                torch.save(
                    sam.state_dict(),
                    os.path.join(out_dir, "best_metric_model.pth"),
                )
                print(f"New best metric found: {metric_handler.best_metric}")

            print(
                f"current epoch: {epoch + 1} current mean dice: {dice_metric:.4f}"
                f"\nbest mean dice: {metric_handler.best_metric:.4f} "
                f"at epoch: {metric_handler.best_epoch}"
            )

        scheduler.step(epoch_loss)

    return epoch_loss_values


def get_epoch_loss(
    optimizer: torch.optim.Optimizer,
    batch_data: dict[str, torch.Tensor],
    sam: Sam,
    loss_function: torch.nn.Module,
    device: str | torch.device,
    pos_sample_points: int,
    neg_sample_points: int,
    use_bboxes: bool,
):
    inputs, labels, patients = (batch_data["image"].to(device), batch_data["label"].to(device), batch_data["patient"])

    sam.train()
    predictions, _, _, _ = get_batch_predictions(
        sam=sam,
        inputs=inputs,
        labels=labels,
        patients=patients,
        pos_sample_points=pos_sample_points,
        neg_sample_points=neg_sample_points,
        use_bboxes=use_bboxes,
    )

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
    metric_handler: MetricHandler,
    pos_sample_points: int,
    neg_sample_points: int,
    use_bboxes: bool,
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
                metric_handler=metric_handler,
                pos_sample_points=pos_sample_points,
                neg_sample_points=neg_sample_points,
                use_bboxes=use_bboxes,
            )

        validation_loss /= step
        wandb.log({"validation_loss": validation_loss})
        print(f"validation_loss: {validation_loss:.4f}")

        metric_handler.aggregate_and_reset_metrics()


def get_validation_loss(
    val_data: dict[str, torch.Tensor],
    sam: Sam,
    loss_function: torch.nn.Module,
    device: str | torch.device,
    metric_handler: MetricHandler,
    pos_sample_points: int,
    neg_sample_points: int,
    use_bboxes: bool,
):
    val_inputs, val_labels, val_patients = (
        val_data["image"].to(device),
        val_data["label"].to(device),
        val_data["patient"],
    )
    val_outputs, _, _, _ = get_batch_predictions(
        sam=sam,
        inputs=val_inputs,
        labels=val_labels,
        patients=val_patients,
        pos_sample_points=pos_sample_points,
        neg_sample_points=neg_sample_points,
        use_bboxes=use_bboxes,
    )

    val_loss = compute_val_loss_and_metrics(
        inputs=val_inputs,
        outputs=val_outputs,
        labels=val_labels,
        loss_function=loss_function,
        metric_handler=metric_handler,
    )
    return val_loss.item()
