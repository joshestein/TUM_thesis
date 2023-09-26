import numpy as np
import torch
import torch.nn.functional as F

from batchgenerators.transforms.abstract_transforms import AbstractTransform

"""
The batch generator functions expect data from a dataloader.
But we are calling the transforms before the data is loaded into the dataloader.
So we need to add a batch dimension to the data, and remove it again after the transforms.
Since we only ever use a batch size of 1, we can do this confidently.
"""


class EnsureBatchDimension(AbstractTransform):
    def __call__(self, **data_dict):
        for key in data_dict.keys():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = data_dict[key][None]

        return data_dict


class RemoveBatchDimension(AbstractTransform):
    def __call__(self, **data_dict):
        for key in data_dict.keys():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = data_dict[key][0]

        return data_dict


class CropOrPad(AbstractTransform):
    """nnUnet does some pretty gnarly padding during 2D data loading. Firstly, this feels to me like it should be a
    transform. Secondly, I don't have time to analyze how it all works - for now I copy the most important bits.

    I tried walking through step-by-step to understand what's happening. All I figured out in the end was we are padding
    the image with zeros to the patch size, and padding the label with -1s.
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, **data_dict):
        target_size = self.patch_size[0]  # TODO: do separate height and width handling

        processed_images, processed_labels = [], []

        for i in range(len(data_dict["image"])):
            image = torch.from_numpy(data_dict["image"][i])
            label = torch.from_numpy(data_dict["label"][i])
            height, width = image.shape[-2], image.shape[-1]
            pad_height = max(0, target_size - height)
            pad_width = max(0, target_size - width)
            if pad_height > 0 or pad_width > 0:
                padding = [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2]
                image = F.pad(image, padding, mode="constant", value=0)
                label = F.pad(label, padding, mode="constant", value=0)

            if image.shape[-2] > target_size or image.shape[-1] > target_size:
                x = (image.shape[-1] - target_size) // 2
                y = (image.shape[-2] - target_size) // 2
                image = image[:, y : y + target_size, x : x + target_size]
                label = label[:, y : y + target_size, x : x + target_size]

            processed_images.append(image.numpy())
            processed_labels.append(label.numpy())

        data_dict["image"] = np.stack(processed_images)
        data_dict["label"] = np.stack(processed_labels)

        return data_dict
