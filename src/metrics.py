from typing import Optional

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


class MetricHandler:
    def __init__(self, metric_to_track: str):
        self.metrics = METRICS
        self.metric_values = {metric: [] for metric in METRICS.keys()}
        self.best_epoch = 0
        self.best_metric = 0
        self.metric_to_track = metric_to_track

    def add_metric(self, name, metric):
        self.metrics[name] = metric
        self.metric_values[name] = []

    def accumulate_metrics(self, y, y_pred):
        """Should be called during validation."""
        for metric in self.metrics.values():
            metric(y, y_pred)

    def aggregate_and_reset_metrics(self):
        """Should be called at the end of the validation epoch."""
        for name, metric in self.metrics.items():
            metric_value = metric.aggregate()
            metric_value = metric_value.item() if len(metric_value) == 1 else metric_value.tolist()
            self.metric_values[name].append(metric_value)
            wandb.log({f"validation_{name}": metric_value})

            metric.reset()

    def check_best_metric(self, epoch):
        assert (
            self.metric_to_track in self.metric_values
        ), f"Metric to track: {self.metric_to_track} not found in metric values"

        metric = self.metric_values[self.metric_to_track][-1]
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            return True

        return False

    def last_value(self, metric_name: Optional[str] = None):
        if metric_name is None:
            metric_name = self.metric_to_track

        assert metric_name in self.metric_values, f"{metric_name} not being tracked."
        return self.metric_values[metric_name][-1]
