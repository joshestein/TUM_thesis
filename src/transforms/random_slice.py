import numpy as np
import torch
from monai.transforms import MapTransform


class RandomSliced(MapTransform):
    def __call__(self, data):
        d = dict(data)

        value = next(iter(d.values()))

        # Find a random slice from only the non-zero slices
        reshaped = value.reshape(-1, value.size(-1))  # Reshape to B * H * W, S
        non_zero_slices = torch.nonzero(torch.any(reshaped != 0, dim=0)).squeeze().tolist()
        random_slice = np.random.choice(non_zero_slices, 1)

        for key in self.key_iterator(d):
            d[key] = d[key][..., random_slice].squeeze(-1)

        return d
