import os, shutil
import numpy as np
import random
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob
import cv2
import tifffile as tif
import monai
import torch
from skimage import exposure,measure,segmentation,morphology
from monai.inferers import sliding_window_inference
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_num_threads(1)  #这个作为docker上交的时候可以注释掉

# 返回图片在hsv通道的均值和方差作为特征
def extract_feature(image_path: str):
    feature = []
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    means_hsv, std_hsv = cv2.meanStdDev(img_hsv)

    feature.append(means_hsv)
    feature.append(std_hsv)
    return feature

# 根据已经得到的中心点坐标，返回对应的类别
def cluster(image_path:str):
    centroids =  np.array([[  3.68690462,   0.49137748, 105.81631272,   0.64073039,   0.63384792, 33.09373723],
    [ 88.30804682  ,64.27745113, 188.7468829 ,  52.08498724,  62.11390647, 65.65143542],
    [ 43.63164599 ,111.92362705 , 50.05808066,  37.31790109,  81.6843576, 64.13352828],
    [-9.23705556e-14 ,2.48689958e-14 ,2.80961654e+01, 4.97379915e-14, 2.84217094e-14, 4.35067791e+01]])
    
    feature = extract_feature(image_path=image_path)
    feature = np.array(feature)
    feature = feature[:,:,0]
    feature = feature.reshape(feature.shape[0]*feature.shape[1])

    distance = np.zeros(len(centroids))

    for i in range(len(centroids)):
        distance[i] = np.sqrt(np.sum((feature - centroids[i]) ** 2))
    label = np.argmin(distance)
    return label

# 用于baseline方法的推理过程
def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def baseline_inference(img_data,model):
  # 定义baseline的模型，用于推理荧光图像
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      roi_size = (256,256)
      sw_batch_size = 1
      model.eval()

      # noramalize the data
      if len(img_data.shape) == 2:
                      img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
      elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
              img_data = img_data[:,:, :3]
      else:
              pass
      pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
      for i in range(3):
              img_channel_i = img_data[:,:,i]
              if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                      pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
      with torch.no_grad():
              test_npy01 = pre_img_data/np.max(pre_img_data)
              test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
              test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
              test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
              test_pred_npy = test_pred_out[0,1].cpu().numpy()
              # convert probability map to binary mask and apply morphological postprocessing
              test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy>0.5),16))
      return test_pred_mask


def cellpose_inference(img,model,chan,diam):
    masks, flows, styles = model.eval(img, 
                                    channels=chan,
                                    diameter=diam,
                                    flow_threshold=0.4,
                                    cellprob_threshold=0
                                    )
    return masks



def main():
  parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False) 
  # Dataset parameters
  parser.add_argument('-i', '--input_path', default='/data112/NeurISP2022-CellSeg/TuningSet', type=str, help='training data path; subfolders: images, labels')
  parser.add_argument("-o", '--output_path', default='/data112/wzy/NIPS/baseline/work_dir/output1', type=str, help='output path')
  parser.add_argument('--model_path', default='/data112/wzy/NIPS/baseline/work_dir/deeplab_transformer_3class', help='path where to save models and segmentation results')
  # Model parameters
  parser.add_argument('--input_size', default=256, type=int, help='segmentation classes')
  args = parser.parse_args()



  tuning_dir = args.input_path
  output_dir = args.output_path
  img_size = (args.input_size,args.input_size)
  logger = io.logger_setup()    #打印日志，可以注释
  cluster_model = []
  for i in range(4):
      model_path = '/data112/wzy/NIPS/data/Train_Pre_3class/cluster/class_{}/models/cellpose_cluster_v{}'.format(i,i)
      model = models.CellposeModel(gpu=True, 
                                  pretrained_model=model_path)
      cluster_model.append(model)

  # 替换掉荧光图像的推理模型
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  baseline_model = monai.networks.nets.SwinUNETR(
                  img_size=img_size, 
                  in_channels=3, 
                  out_channels=3,
                  feature_size=24, # should be divisible by 12
                  spatial_dims=2
                  ).to(device)
  checkpoint = torch.load(os.path.join('/data112/wzy/NIPS/baseline/work_dir/deeplab_transformer_3class', 'best_Dice_model2_0.7494.pth'), map_location=torch.device(device))
  baseline_model.load_state_dict(checkpoint['model_state_dict'])
  cluster_model[2] = baseline_model


  imgs_path = os.listdir(tuning_dir)
  imgs_path.sort()
  # imgs_path = imgs_path[0:-1]         # 那张wsi不能这么处理
  chan = [[2,0],[2,0],[3,0],[2,0]]
  diam = [45,45,15,45]
  for i,img_path in enumerate(imgs_path):
      img = io.imread(os.path.join(tuning_dir,img_path))
      img_name = img_path.split('.')[0]
      print(img_name)
      label = cluster(os.path.join(tuning_dir,img_path))
      print(label)
      if(label == 2):
            masks = baseline_inference(img,cluster_model[label])
      else:
            masks = cellpose_inference(img,cluster_model[label],chan[label],diam[label])
            masks_reverse = cellpose_inference(255-img,cluster_model[label],chan[label],diam[label])
            a = len(np.unique(masks,return_index=False,return_counts=True,return_inverse=False)[0])
            b = len(np.unique(masks_reverse,return_index=False,return_counts=True,return_inverse=False)[0])
            if(a < b):masks = masks_reverse

            if(len(np.unique(masks,return_index=False,return_counts=True,return_inverse=False)[0])<= 5):
                masks = baseline_inference(img,cluster_model[2])
      tif.imwrite(os.path.join(output_dir,(img_name+'_label.tiff')),masks,compression='zlib')


if __name__ == "__main__":
    main()