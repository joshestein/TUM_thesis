from typing import Hashable, Mapping

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


class RemoveSlicesd(MapTransform):
    """Remove slices from the data. Expects the number of slices to be the last dimension of the data.

    :param keys: Keys to remove slices from.
    :param percentage_slices: Percentage of slices to keep from the data.
    :param random_slices: Whether to remove random slices or slices from the center of the volume. Defaults to `True`.
    :param maintain_shape: Whether to maintain the shape of the data. If `True`, the removed slices will be filled with
            zeros. Defaults to `True`.
    :param allow_missing_keys: Allow missing keys. If `False`, an exception will be raised if a key is missing. Defaults
            to `True`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        percentage_slices: float,
        random_slices=True,
        maintain_shape=True,
        allow_missing_keys=True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.keys = keys
        self.percentage_slices = percentage_slices
        self.random_slices = random_slices
        self.maintain_shape = maintain_shape

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        if self.percentage_slices == 1.0:
            return d

        value = next(iter(d.values()))
        indexes = self._get_indexes(value)

        for key in self.key_iterator(d):
            if self.maintain_shape:
                d[key][..., indexes] = 0.0
            else:
                d[key] = d[key][..., indexes]

        return d

    def _get_indexes(self, data: torch.Tensor):
        slices = data.shape[-1]
        # We subtract from one to ensure this is the amount we keep.
        # For example, if 0.8, we want to keep 80% of the data, so we get indexes corresponding to the other 20%.
        num_slices = int(slices * (1 - self.percentage_slices))

        if self.random_slices:
            indexes = torch.randperm(slices)[:num_slices]
            return indexes

        start_slice = slices // 2 - num_slices // 2
        end_slice = start_slice + num_slices
        return torch.arange(start_slice, end_slice)
