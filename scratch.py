from PIL import Image
from backbone import reconstruct_image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loader import get_celeba_dataset
from backbone import SkipNet, SfsNetPipeline
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from loader import wandb_log_images
import wandb
import os

def test(model_dir=None, img_path=None, out_dir=None, lighting_pos=None):
    base_model = SkipNet()
    base_model.load_state_dict(torch.load(model_dir))
    transform_one = transforms.Compose([
        transforms.Resize(128),
    ])
    transform_two = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform_one(Image.open(img_path))
    img = np.array(img)
    img = transform_two(img)
    img = torch.reshape(img, [1,3,128,128])
    suffix = 'test'
    predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = base_model(img)
    wandb.init(tensorboard=True)
    wandb_log_images(wandb, predicted_normal, None, suffix + "Predicted Normal", \
                     0, suffix + "Predicted Normal", path= out_dir + "_predicted_normal_new.png")
    wandb_log_images(wandb, predicted_albedo, None, suffix + "Predicted Alebdo", \
                     0, suffix + "Predicted Albedo", path= out_dir + "_predicted albedo_new.png")
    wandb_log_images(wandb, out_shading, None, suffix + "Out Shading", \
                     0, suffix + "Out Shading", path= out_dir + "_out shading_new.png")
    wandb_log_images(wandb, out_recon, None, suffix + "Out Recon", \
                     0, suffix + "Out Recon", path= out_dir + "reconstruction_-0.5.png")

if __name__ == '__main__':
    model_dir = '/home/hd8t/xiangyu.yin/results/metadata/checkpoints/Skip_First.pkl'
    img_path = '/home/hd8t/xiangyu.yin/timg.jpg'
    out_dir = '/home/hd8t/xiangyu.yin/results/metadata/out_images/Lighting_27/'
    test(model_dir, img_path, out_dir)



