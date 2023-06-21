from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    SpatialPadd,
    ToTensord,
    Transposed,
)
from monai.utils import InterpolateMode

from src.transforms.random_slice import RandomSliced
from src.transforms.remove_slices import RemoveSlicesd


def get_transforms(
    spatial_dims: int,
    augment: bool = True,
    percentage_slices: float = 1.0,
    sample_regions=("apex", "base", "mid"),
) -> (Compose, Compose):
    # TODO: once we start working with the full volume (not just ED and ES) we will need to update the transforms

    keys = ["image", "label"]

    train_transforms = [
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], channel_wise=False),
        # NormalizeIntensityd(keys=["image"], channel_wise=True),
        # Spacingd(keys=keys, pixdim=(1.25, 1.25, -1.0), mode=("nearest", "nearest")),
    ]

    val_transforms = [
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], channel_wise=False),
    ]

    if spatial_dims == 3:
        spatial_size = (224, 224, 16)
        transposition_indices = (0, 3, 1, 2)
    else:
        spatial_size = (224, 224)
        transposition_indices = (0, 1, 2)
        train_transforms += [RandomSliced(keys=keys)]
        val_transforms += [RandomSliced(keys=keys)]

    if augment:
        train_transforms += [
            RandAdjustContrastd(keys=["image"], gamma=(0.8, 1.2), prob=0.5),
            RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
            RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
            RandZoomd(
                keys=keys,
                min_zoom=0.85,
                max_zoom=1.15,
                # Use area interpolation for images and nearest neighbor for labels.
                mode=(InterpolateMode.AREA, InterpolateMode.NEAREST_EXACT),
                prob=0.25,
            ),
            RandRotate90d(keys=keys, spatial_axes=(0, 1), prob=0.25),
        ]

    train_transforms += [
        RemoveSlicesd(
            keys=keys,
            percentage_slices=percentage_slices,
            sample_regions=sample_regions,
        ),
        # ResizeWithPadOrCrop only center crops - we want random cropping, so we explicitly pad and then crop
        # Since we have 4 layers in UNet, we must have dimensions divisible by 2**4 = 16
        SpatialPadd(keys=keys, spatial_size=spatial_size, mode="constant"),
        RandSpatialCropd(keys=keys, roi_size=spatial_size, random_size=False),
        # Move depth to the second dimension (Pytorch expects 3D inputs in the shape of C x D x H x W)
        Transposed(keys=keys, indices=transposition_indices),
        ToTensord(keys=keys),
    ]

    val_transforms += [
        ResizeWithPadOrCropd(keys=keys, spatial_size=spatial_size, mode="constant"),
        Transposed(keys=keys, indices=transposition_indices),
        ToTensord(keys=keys),
    ]

    return Compose(train_transforms), Compose(val_transforms)
