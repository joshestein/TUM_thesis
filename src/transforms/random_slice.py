import torch
from monai.transforms import MapTransform


class RandomSliced(MapTransform):
    def __call__(self, data):
        d = dict(data)

        value = next(iter(d.values()))
        num_slices = value.shape[-1]
        random_slice = torch.randint(num_slices, size=(1,))

        for key in self.key_iterator(d):
            d[key] = d[key][..., random_slice].squeeze(-1)

        return d
