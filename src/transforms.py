from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    DivisiblePadd,
    NormalizeIntensityd,
    ToTensord,
    RandAdjustContrastd,
    RandRotated,
    RandZoomd,
    RandFlipd,
)
from monai.utils import InterpolateMode


def get_transforms(
    image_keys: list[str] = None,
    label_keys: list[str] = None,
):
    if image_keys is None:
        image_keys = ["end_diastole", "end_systole"]

    if label_keys is None:
        label_keys = ["end_diastole_label", "end_systole_label"]

    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=[*image_keys, *label_keys], channel_dim="no_channel"),
            Orientationd(keys=[*image_keys, *label_keys], axcodes="RAS"),
            DivisiblePadd(keys=[*image_keys, *label_keys], k=16, mode="reflect"),
            RandRotated(
                keys=[*image_keys, *label_keys], range_x=0.52, range_y=0.52, range_z=0.52, prob=0.5
            ),  # 30 degrees
            RandAdjustContrastd(keys=[*image_keys], gamma=(0.7, 1.5), prob=0.5),
            RandFlipd(keys=[*image_keys, *label_keys], spatial_axis=[0, 1, 2], prob=0.5),
            RandZoomd(
                keys=[*image_keys, *label_keys], min_zoom=0.85, max_zoom=1.25, mode=InterpolateMode.NEAREST, prob=0.5
            ),
            SpatialPadd(keys=[*image_keys, *label_keys], spatial_size=(224, 224, 16)),
            RandSpatialCropd(keys=[*image_keys, *label_keys], roi_size=(224, 224, 16), random_size=False),
            # ResizeWithPadOrCrop(keys=[*image_keys, *label_keys], spatial_size=(224, 224, 16)),
            NormalizeIntensityd(keys=[*image_keys], channel_wise=True),
            ToTensord(keys=[*image_keys, *label_keys]),
        ]
    )

    return train_transforms
