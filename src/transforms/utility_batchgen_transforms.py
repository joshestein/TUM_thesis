import numpy as np
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
