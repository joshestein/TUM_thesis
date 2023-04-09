from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
    Transposed,
)
from monai.utils import InterpolateMode


def get_transforms(
    augment: bool = True,
    image_keys: list[str] = None,
    label_keys: list[str] = None,
):
    # TODO: once we start working with the full volume (not just ED and ES) we will need to update the transforms

    if image_keys is None:
        image_keys = ["end_diastole", "end_systole"]

    if label_keys is None:
        label_keys = ["end_diastole_label", "end_systole_label"]

    train_transforms = [
        EnsureChannelFirstd(keys=[*image_keys, *label_keys], channel_dim="no_channel"),
        NormalizeIntensityd(keys=[*image_keys], channel_wise=True),
        # Spacingd(keys=[*image_keys, *label_keys], pixdim=(1.25, 1.25, 10.0)),
    ]

    if augment:
        # Use trilinear interpolation for images and nearest neighbor for labels.
        interpolation_keys = [InterpolateMode.TRILINEAR] * len(image_keys), [InterpolateMode.NEAREST] * len(label_keys)
        train_transforms += [
            RandAdjustContrastd(keys=[*image_keys], gamma=(0.8, 1.2), prob=0.5),
            RandFlipd(keys=[*image_keys, *label_keys], spatial_axis=0, prob=0.5),
            RandFlipd(keys=[*image_keys, *label_keys], spatial_axis=1, prob=0.5),
            RandZoomd(
                keys=[*image_keys, *label_keys],
                min_zoom=0.85,
                max_zoom=1.15,
                mode=[key for nested_list in interpolation_keys for key in nested_list],  # flatten nested tuple
                prob=0.5,
            ),
        ]

    train_transforms += [
        # ResizeWithPadOrCrop only center crops - we want random cropping, so we explicitly pad and then crop
        # Since we have 4 layers in UNet, we must have dimensions divisible by 2**4 = 16
        SpatialPadd(keys=[*image_keys, *label_keys], spatial_size=(224, 224, 16), mode="reflect"),
        RandSpatialCropd(keys=[*image_keys, *label_keys], roi_size=(224, 224, 16), random_size=False),
        # Move depth to the second dimension (Pytorch expects 3D inputs in the shape of C x D x H x W)
        Transposed(keys=[*image_keys, *label_keys], indices=(0, 3, 1, 2)),
        ToTensord(keys=[*image_keys, *label_keys]),
    ]

    return Compose(train_transforms)
