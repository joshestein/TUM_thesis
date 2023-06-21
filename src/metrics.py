from monai.metrics import DiceMetric, HausdorffDistanceMetric

METRICS = {
    "dice": DiceMetric(include_background=False, reduction="mean"),
    "dice_with_background": DiceMetric(include_background=True, reduction="mean"),
    "dice_per_label": DiceMetric(include_background=False, reduction="mean_batch"),
    "dice_per_label_with_background": DiceMetric(include_background=True, reduction="mean_batch"),
    "hausdorff": HausdorffDistanceMetric(include_background=False, reduction="mean"),
    "hausdorff_per_label": HausdorffDistanceMetric(include_background=False, reduction="mean_batch"),
}
