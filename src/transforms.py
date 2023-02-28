from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, DivisiblePadd, NormalizeIntensityd, ToTensord


def get_transforms(image_key: str, label_key: str):
    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=[image_key, label_key], channel_dim="no_channel"),
            Orientationd(keys=[image_key, label_key], axcodes="RAS"),
            DivisiblePadd(keys=[image_key, label_key], k=16, mode="reflect"),
            NormalizeIntensityd(keys=[image_key], channel_wise=True),
            ToTensord(keys=[image_key, label_key]),
        ]
    )
    return train_transforms
