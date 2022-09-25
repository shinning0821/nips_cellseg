from turtle import forward
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeeplabV3(nn.Module):
    def __init__(self, num_classes, encoder_name) -> None:
        super().__init__()
        self.net = smp.DeepLabV3(encoder_name=encoder_name,
                            classes=num_classes)

    def forward(self, x):
        x = self.net(x)
        return x
