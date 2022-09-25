import torch
import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
import numpy as np
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


post_pred = Compose(
    [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
)
post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])

def compute_DiceMetric(args,device,val_loader,model):
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    val_images = None
    val_labels = None
    val_outputs = None
    for val_data in val_loader:
        val_images, val_labels = val_data["img"].to(device), val_data[
            "label"
        ].to(device)
        val_labels_onehot = monai.networks.one_hot(
            val_labels, args.num_class
        )
        roi_size = (256, 256)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_images, roi_size, sw_batch_size, model
        )  
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels_onehot = [
            post_gt(i) for i in decollate_batch(val_labels_onehot)
        ]
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels_onehot)
    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric,val_images,val_labels,val_outputs



# def compute_F1Metric(args,device,val_loader,model):
#     f1_metric = F1Metric(
#         include_background=False, reduction="mean", get_not_nans=False
#     )
#     val_images = None
#     val_labels = None
#     val_outputs = None
#     for val_data in val_loader:
#         val_images, val_labels = val_data["img"].to(device), val_data[
#             "label"
#         ].to(device)
#         val_labels_onehot = monai.networks.one_hot(
#             val_labels, args.num_class
#         )
#         roi_size = (256, 256)
#         sw_batch_size = 4
#         val_outputs = sliding_window_inference(
#             val_images, roi_size, sw_batch_size, model
#         )  
#         val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
#         val_labels_onehot = [
#             post_gt(i) for i in decollate_batch(val_labels_onehot)
#         ]
#         # compute metric for current iteration
#         f1_metric(y_pred=val_outputs, y=val_labels_onehot)
#     # aggregate the final mean dice result
#     metric = f1_metric.aggregate().item()
#     f1_metric.reset()
#     return metric,val_images,val_labels,val_outputs