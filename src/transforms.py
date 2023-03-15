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


def get_transforms(image_key: str, label_key: str):
    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=[image_key, label_key], channel_dim="no_channel"),
            Orientationd(keys=[image_key, label_key], axcodes="RAS"),
            DivisiblePadd(keys=[image_key, label_key], k=16, mode="reflect"),
            RandRotated(keys=[image_key, label_key], range_x=0.52, range_y=0.52, range_z=0.52, prob=0.5),  # 30 degrees
            RandAdjustContrastd(keys=[image_key], gamma=(0.7, 1.5), prob=0.5),
            RandFlipd(keys=[image_key, label_key], spatial_axis=[0, 1, 2], prob=0.5),
            RandZoomd(
                keys=[image_key, label_key], min_zoom=0.85, max_zoom=1.25, mode=InterpolateMode.NEAREST, prob=0.5
            ),
            NormalizeIntensityd(keys=[image_key], channel_wise=True),
            ToTensord(keys=[image_key, label_key]),
        ]
    )
    return train_transforms
