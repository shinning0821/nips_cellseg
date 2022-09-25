from turtle import forward
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
config_file = '/data112/wzy/mmsegmentation-master/configs/danet/danet_r101-d8_512x512_80k_ade20k.py'
checkpoint_file = '/data112/wzy/mmsegmentation-master/result/result16/epoch_60.pth'
 
# 从一个 config 配置文件和 checkpoint 文件里创建分割模型


class DAnet(nn.Module):
    def __init__(self, num_classes, encoder_name) -> None:
        super().__init__()
        model = init_segmentor(config_file,checkpoint_file,device = 'cuda:6')
        model.decode_head.num_classes = num_classes
        self.net = model

    def forward(self, x):
        return self.net.forward_dummy(x)
