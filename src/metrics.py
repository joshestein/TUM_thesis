import wandb
from monai.metrics import DiceMetric, HausdorffDistanceMetric

METRICS = {
    "dice": DiceMetric(include_background=False, reduction="mean", num_classes=4),
    "dice_with_background": DiceMetric(include_background=True, reduction="mean", num_classes=4),
    "dice_per_label": DiceMetric(include_background=False, reduction="mean_batch", num_classes=4),
    "dice_per_label_with_background": DiceMetric(include_background=True, reduction="mean_batch", num_classes=4),
    "hausdorff": HausdorffDistanceMetric(include_background=False, reduction="mean"),
    "hausdorff_per_label": HausdorffDistanceMetric(include_background=False, reduction="mean_batch"),
}


def aggregate_validation_metrics(metric_values: dict[str, list[float] | list[list[float]]]):
    """Aggregates and resets and metrics. This should be called at the end of each validation epoch."""
    for name, metric in METRICS.items():
        # aggregate and reset metrics
        metric_value = metric.aggregate()
        metric_value = metric_value.item() if len(metric_value) == 1 else metric_value.tolist()
        metric.reset()

        if name not in metric_values:
            metric_values[name] = []

        metric_values[name].append(metric_value)
        wandb.log({f"validation_{name}": metric_value})
