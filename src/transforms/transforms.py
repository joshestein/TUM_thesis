from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    ToTensord,
    Transposed,
)
from monai.utils import InterpolateMode

from src.transforms.remove_slices import RemoveSlicesd


def get_transforms(
    augment: bool = True,
    num_slices: int | None = None,
    sample_regions=("apex", "base", "mid"),
) -> (Compose, Compose):
    # TODO: once we start working with the full volume (not just ED and ES) we will need to update the transforms

    keys = ["image", "label"]

    # Spacingd(keys=keys, pixdim=(1.25, 1.25, -1.0), mode=("nearest", "nearest")),
    train_transforms = [EnsureChannelFirstd(keys=keys, channel_dim="no_channel")]
    val_transforms = [EnsureChannelFirstd(keys=keys, channel_dim="no_channel")]

    # Pad to fixed size
    spatial_size = (224, 224, 16)
    train_transforms += [
        ResizeWithPadOrCropd(keys=keys, spatial_size=spatial_size, mode="constant"),
        RemoveSlicesd(
            keys=keys,
            num_slices=num_slices,
            sample_regions=sample_regions,
        ),
    ]

    val_transforms += [
        ResizeWithPadOrCropd(keys=keys, spatial_size=spatial_size, mode="constant"),
    ]

    if augment:
        train_transforms += [
            RandAdjustContrastd(keys=["image"], gamma=(0.8, 1.2), prob=0.5),
            RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
            RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
            RandZoomd(
                keys=keys,
                min_zoom=0.85,
                max_zoom=1.15,
                # Use area interpolation for images and nearest neighbour for labels.
                mode=(InterpolateMode.AREA, InterpolateMode.NEAREST_EXACT),
                prob=0.25,
            ),
            RandRotate90d(keys=keys, spatial_axes=(0, 1), prob=0.25),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
        ]

    # transposition_indices = (0, 1, 2) if spatial_dims == 2 else (0, 2, 3, 1)

    train_transforms += [
        # Move depth to the second dimension (Pytorch expects 3D inputs in the shape of C x D x H x W)
        # Transposed(keys=keys, indices=transposition_indices),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        # ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=keys),
    ]

    val_transforms += [
        # Transposed(keys=keys, indices=transposition_indices),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        # ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=keys),
    ]

    return Compose(train_transforms), Compose(val_transforms)
