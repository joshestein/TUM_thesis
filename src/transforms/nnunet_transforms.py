import math

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform, RenameTransform

from src.transforms.utility_batchgen_transforms import EnsureBatchDimension, RemoveBatchDimension


def get_nnunet_transforms():
    # These values are just extracted by inspecting the nnUNet pipeline
    # This is for 2D ACDC
    # TODO: check for MNMs
    patch_size_spatial = [224, 224]
    rotation_for_DA = {"x": (-math.pi, math.pi), "y": (0, 0), "z": (0, 0)}
    order_resampling_data = 3
    order_resampling_seg = 1
    border_val_seg = -1

    train_transforms = [
        EnsureBatchDimension(),
        RenameTransform("image", "data", True),
        RenameTransform("label", "seg", True),
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0, 0),
            sigma=(0, 0),
            do_rotation=True,
            angle_x=rotation_for_DA["x"],
            angle_y=rotation_for_DA["y"],
            angle_z=rotation_for_DA["z"],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True,
            scale=(0.7, 1.4),
            border_mode_data="constant",
            border_cval_data=0,
            order_data=order_resampling_data,
            border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0,
            p_scale_per_sample=0.2,
            p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,  # todo experiment with this
        ),
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        ContrastAugmentationTransform(p_per_sample=0.15),
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=None,
        ),
        GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),
        GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3),
    ]

    mirror_axes = (0, 1)
    if mirror_axes is not None and len(mirror_axes) > 0:
        train_transforms.append(MirrorTransform(mirror_axes))

    train_transforms.append(RemoveLabelTransform(-1, 0))

    # if deep_supervision_scales is not None:
    #     tr_transforms.append(
    #         DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key="target", output_key="target")
    #     )
    #
    train_transforms += [
        RenameTransform("data", "image", True),
        RenameTransform("seg", "label", True),
        RemoveBatchDimension(),
        NumpyToTensor(["image", "label"], "float"),
    ]

    val_transforms = [
        EnsureBatchDimension(),
        RemoveLabelTransform(-1, 0, input_key="label", output_key="label"),
        RemoveBatchDimension(),
        NumpyToTensor(["image", "label"], "float"),
    ]

    return Compose(train_transforms), Compose(val_transforms)
