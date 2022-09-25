import numpy as np
import torch
import monai
from torch.utils.data import DataLoader
from monai.data import decollate_batch, PILReader
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
)
#%% define transforms for image and segmentation


def return_trainloader(args,train_files):
    train_transforms = Compose(
    [
        LoadImaged(
            keys=["img", "label"], reader=PILReader, dtype=np.uint8
        ),  # image three channels (H, W, 3); label: (H, W)
        AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
        AsChannelFirstd(
            keys=["img"], channel_dim=-1, allow_missing_keys=True
        ),  # image: (3, H, W)
        ScaleIntensityd(
            keys=["img"], allow_missing_keys=True
        ),  # Do not scale label
        SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
        RandSpatialCropd(
            keys=["img", "label"], roi_size=args.input_size, random_size=False
        ),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        # # intensity transform
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandZoomd(
            keys=["img", "label"],
            prob=0.15,
            min_zoom=0.8,
            max_zoom=1.5,
            mode=["area", "nearest"],
        ),
        EnsureTyped(keys=["img", "label"]),
        ]
    )
    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return train_loader


def return_valloader(args,val_files):
    val_transforms = Compose(
    [
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        AddChanneld(keys=["label"], allow_missing_keys=True),
        AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        # AsDiscreted(keys=['label'], to_onehot=3),
        EnsureTyped(keys=["img", "label"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    return val_loader

def return_unlableloader(args,unlable_files):
    unlable_transforms = Compose(
    [
        LoadImaged(
            keys=["img"], reader=PILReader, dtype=np.uint8
        ),  # image three channels (H, W, 3); label: (H, W)
        AsChannelFirstd(
            keys=["img"], channel_dim=-1, allow_missing_keys=True
        ),  # image: (3, H, W)
        ScaleIntensityd(
            keys=["img"], allow_missing_keys=True
        ),  # Do not scale label
        SpatialPadd(keys=["img"], spatial_size=args.input_size),
        RandSpatialCropd(
            keys=["img"], roi_size=args.input_size, random_size=False
        ),
        RandAxisFlipd(keys=["img"], prob=0.5),
        RandRotate90d(keys=["img"], prob=0.5, spatial_axes=[0, 1]),
        # # intensity transform
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        EnsureTyped(keys=["img"]),
        ]
    )
    unlable_ds = monai.data.Dataset(data=unlable_files, transform=unlable_transforms)
    unlable_loader = DataLoader(
        unlable_ds,
        batch_size=2*args.batch_size ,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return unlable_loader