from typing import Optional

import numpy as np
import surface_distance
import wandb
from monai.metrics import DiceMetric, HausdorffDistanceMetric

METRICS = {
    "dice": DiceMetric(include_background=False, reduction="mean", num_classes=4),
    "dice_with_background": DiceMetric(include_background=True, reduction="mean", num_classes=4),
    "dice_per_label": DiceMetric(include_background=False, reduction="mean_batch", num_classes=4),
    "dice_per_label_with_background": DiceMetric(include_background=True, reduction="mean_batch", num_classes=4),
    "hausdorff": HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=0.95),
    "hausdorff_per_label": HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=0.95),
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


def compute_surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: list[float],
    hausdorff_percentile=95,
):
    # Ignore background class
    prediction = prediction[:, 1:]
    target = target[:, 1:]

    sd = np.empty((prediction.shape[0], prediction.shape[1]))
    hd = np.empty((prediction.shape[0], prediction.shape[1]))
    mad = np.empty((prediction.shape[0], prediction.shape[1]))

    for batch_index, class_index in np.ndindex(prediction.shape[0], prediction.shape[1]):
        metrics = compute_np_surface_metrics(
            # Convert to boolean arrays
            prediction[batch_index, class_index] == 1,
            target[batch_index, class_index] == 1,
            spacing_mm=spacing_mm,
            hausdorff_percentile=hausdorff_percentile,
        )
        hd[batch_index, class_index] = metrics["hausdorff"]
        sd[batch_index, class_index] = metrics["surface_distance"]
        mad[batch_index, class_index] = metrics["mean_absolute_difference"]

    # Average over batch
    return {
        "avg_surface_distance": np.nanmean(sd, axis=0),
        "hausdorff": np.nanmean(hd, axis=0),
        "mean_absolute_difference": np.nanmean(mad, axis=0),
    }


def compute_np_surface_metrics(pred: np.ndarray, target: np.ndarray, spacing_mm: list[float], hausdorff_percentile=95):
    surface_distances = surface_distance.compute_surface_distances(pred, target, spacing_mm=spacing_mm)

    hd = surface_distance.compute_robust_hausdorff(surface_distances, hausdorff_percentile)
    dist_gt_to_prediction, dist_prediction_to_gt = surface_distance.compute_average_surface_distance(surface_distances)
    sd = max(dist_gt_to_prediction, dist_prediction_to_gt)
    absolute_difference = surface_distances["mean_absolute_difference"]

    # Replace infs with NaNs
    # This allows us to average using np.nanmean
    sd = np.nan if sd == np.inf else sd
    hd = np.nan if hd == np.inf else hd
    absolute_difference = np.nan if absolute_difference == np.inf else absolute_difference

    return {
        "surface_distance": sd,
        "hausdorff": hd,
        "mean_absolute_difference": absolute_difference,
    }
