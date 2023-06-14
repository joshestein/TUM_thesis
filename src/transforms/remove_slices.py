import math
from typing import Hashable, Mapping

import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform


class RemoveSlicesd(MapTransform):
    """Remove slices from the data. Expects the number of slices to be the last dimension of the data. This transform
    should always be called before any padding functions.

    :param keys: Keys to remove slices from.
    :param percentage_slices: Percentage of slices to keep from the data.
    :param random_slices: Whether to remove random slices or slices from the center of the volume. Defaults to `True`.
    :param maintain_shape: Whether to maintain the shape of the data. If `True`, the removed slices will be filled with
            zeros. Defaults to `True`.
    :param sample_regions: Which region(s) of the heart to sample from. Can be one or more of 'apex', 'mid', 'base'.
        Defaults to ('apex', 'mid', 'base').
    :param allow_missing_keys: Allow missing keys. If `False`, an exception will be raised if a key is missing. Defaults
            to `True`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        percentage_slices: float,
        random_slices=True,
        maintain_shape=True,
        sample_regions: list[str] = ("apex", "mid", "base"),
        allow_missing_keys=True,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.keys = keys
        self.percentage_slices = percentage_slices
        self.random_slices = random_slices
        self.maintain_shape = maintain_shape

        for region in sample_regions:
            assert region in ("apex", "mid", "base"), f"Invalid region {region}. Must be one of 'apex', 'mid', 'base'."
        self.sample_regions = sample_regions

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        if self.percentage_slices == 1.0 and len(self.sample_regions) == 3:
            return d

        value = next(iter(d.values()))
        mask = self._get_mask(value)

        for key in self.key_iterator(d):
            if self.maintain_shape:
                # Zero out everything except the mask
                d[key][..., ~mask] = 0.0
            else:
                # Keep only the data in the mask
                d[key] = d[key][..., mask]

        return d

    def _get_mask(self, data: torch.Tensor):
        slices = data.shape[-1]
        slices_per_region = slices / 3  # We divide the entire volume into 3 regions: base, mid, apex
        num_sample_slices = int(slices * self.percentage_slices)

        region_slices = {
            "base": range(0, int(math.ceil(slices_per_region))),
            "mid": range(int(math.ceil(slices_per_region)), int(math.ceil(2 * slices_per_region))),
            "apex": range(int(math.ceil(2 * slices_per_region)), slices),
        }

        if self.random_slices:
            indices_to_sample = [index for region in self.sample_regions for index in region_slices[region]]
            if len(indices_to_sample) < num_sample_slices:
                print("Warning: Not enough slices to sample from. Using all slices in the sample regions.")
                indices = indices_to_sample
            else:
                indices = torch.randperm(len(indices_to_sample))[:num_sample_slices]
        else:
            samples_per_region = int(num_sample_slices / len(self.sample_regions))
            indices = []
            for region in self.sample_regions:
                start = region_slices[region][0]
                end = region_slices[region][-1]

                if end - start < samples_per_region:
                    print(f"Warning: Too few slices in region {region} - using all slices.")
                    indices.extend(region_slices[region])
                else:
                    mid = int((start + end) / 2)
                    start = mid - int(samples_per_region / 2)
                    end = mid + int(samples_per_region / 2)
                    indices.extend(range(start, end))

        mask = torch.zeros(slices, dtype=torch.bool)
        mask[indices] = True
        return mask
