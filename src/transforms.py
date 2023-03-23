from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotated,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)
from monai.utils import InterpolateMode


def get_transforms(
    augment: bool = True,
    image_keys: list[str] = None,
    label_keys: list[str] = None,
):
    if image_keys is None:
        image_keys = ["end_diastole", "end_systole"]

    if label_keys is None:
        label_keys = ["end_diastole_label", "end_systole_label"]

    train_transforms = [
        EnsureChannelFirstd(keys=[*image_keys, *label_keys], channel_dim="no_channel"),
        Orientationd(keys=[*image_keys, *label_keys], axcodes="RAS"),
        # Since we have 4 layers in UNet, we must have dimensions divisible by 2**4 = 16
        DivisiblePadd(keys=[*image_keys, *label_keys], k=16, mode="reflect"),
    ]

    if augment:
        train_transforms += [
            RandRotated(
                keys=[*image_keys, *label_keys], range_x=0.52, range_y=0.52, range_z=0.52, prob=0.5
            ),  # 30 degrees
            RandAdjustContrastd(keys=[*image_keys], gamma=(0.7, 1.5), prob=0.5),
            RandFlipd(keys=[*image_keys, *label_keys], spatial_axis=0, prob=0.5),
            RandFlipd(keys=[*image_keys, *label_keys], spatial_axis=1, prob=0.5),
            RandZoomd(
                keys=[*image_keys, *label_keys], min_zoom=0.85, max_zoom=1.25, mode=InterpolateMode.NEAREST, prob=0.5
            ),
            SpatialPadd(keys=[*image_keys, *label_keys], spatial_size=(224, 224, 16)),
            RandSpatialCropd(keys=[*image_keys, *label_keys], roi_size=(224, 224, 16), random_size=False),
            # ResizeWithPadOrCrop only center crops - we want random cropping, so we explicitly pad and then crop
        ]

    train_transforms += [
        NormalizeIntensityd(keys=[*image_keys], channel_wise=True),
        ToTensord(keys=[*image_keys, *label_keys]),
    ]

    return Compose(train_transforms)
