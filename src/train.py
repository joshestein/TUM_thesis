import os

import torch
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
    metric: CumulativeIterationMetric,
    device: str | torch.device,
    out_dir: str | os.PathLike,
):
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(to_onehot=4, argmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=4)])

    early_stopper = EarlyStopper(patience=10)

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["end_diastole"].to(device),
                batch_data["end_diastole_label"].to(device),
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        # wandb.log({"epoch_loss": epoch_loss})
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["end_diastole"].to(device),
                        val_data["end_diastole_label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = metric.aggregate().item()
                # reset the status for next validation round
                metric.reset()

                early_stopper.check_early_stop(metric)
                metric_values.append(metric)
                # wandb.log({"validation_dice": metric})

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(out_dir, "best_metric_model.pth"),
                    )
                    print(f"New best metric found: {best_metric}")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

        if early_stopper.stop:
            print("Early stop")
            break

    return epoch_loss_values, metric_values