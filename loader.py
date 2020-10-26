import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

import numpy as np
import cv2
import tqdm

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision

import glob
from random import randint
from PIL import Image
import pandas as pd
from skimage import io
Size_for_image = 128

def apply_mask(input_image, mask):
    if mask is None:
        return input_image
    return input_image * mask

def wandb_log_images(wandb, img, mask, caption, step, log_name, path=None, denormalize=False):
    ndarr = get_image_grid(img, denormalize=denormalize, mask=mask)

    #save image if path is provided
    if path is not None:
        im = Image.fromarray(ndarr)
        im.save(path)

    wimg = wandb.Image(ndarr, caption=caption)
    wandb.log({log_name:wimg})

def save_image(pic, denormalize=False, path=None, mask=None):

    ndarr = get_image_grid(pic, denormalize=denormalize, mask=mask)

    if path == None:
        plt.imshow(ndarr)
        plt.show()

    else:
        img = Image.fromarray(ndarr)
        img.save(path)

def de_norm(x):
    #x here means the input image.
    out = (x + 1) /2
    return out.clamp(0, 1)

#pic here indicates all the pictures
def get_image_grid(pic, denormalize=False, mask=None):
    if denormalize:
        pic = de_norm(pic)

    if mask is not None:
        pic = pic*mask

    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1,2,0).cpu().numpy()
    return ndarr

def get_normal_in_range(normal):
    new_normal = normal * 128 + 128
    new_normal = new_normal.clamp(0, 255) / 255
    return new_normal

def generate_dataset_csv(dir, save_loc):
    albedo = set()
    normal = set()
    depth = set()
    sh = set()
    face = set()
    mask = set()

    name_to_set = {"albedo":albedo,
                   "normal":normal,
                   "depth":depth,
                   "sh":sh,
                   "face":face,
                   "mask":mask}

    for key,value in name_to_set.items():
        for sub_dir in os.listdir(dir):
            match_str = sub_dir + "/*_" + key + "_*"
            for img in sorted(glob(dir + match_str)):
                splited_img_path = img.split("/")
                folder_id = splited_img_path[-2]
                img_name = splited_img_path[-1].split('.')[0].split('_')
                assert (len(img_name) == 4), "The image has wrong name representation"
                img_name = folder_id + "_" + img_name[0] + "_" + img_name[2] + "_" + img_name[3]
                value.add(img_name)

    final_images= set.intersection(albedo, normal, depth, face, mask)

    albedo = []
    normal = []
    depth = []
    sh = []
    face = []
    mask = []
    name = []

    name_to_list = {'albedo':albedo, 'normal':normal, 'depth':depth, "sh": sh, "face": face, "mask": mask, 'name':name}

    for img in final_images:
        split_final_image = img.split('_')
        for key, value in name_to_list.items():
            extension = '.png'
            if key == 'sh':
                extension = '.txt'

            if key == 'name':
                filename = split_final_image[0] + "_" + split_final_image[1] + "_" + key + "_" + "_".join(split_final_image[2:])
            else:
                filename = split_final_image[0] + "_" + split_final_image[1] + "_" + key + "_" + "_".join(split_final_image[2:]) + extension

            value.append(filename)
            print("successfully append the value.{}".format(filename))

    df = pd.DataFrame(data = name_to_list)
    df.to_csv(save_loc)
    print('saved')

def get_celeba_dataset(celeb_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0,
                       training_syn=False):

    face = []
    if read_from_csv is None:
        for img in sorted(glob.glob(celeb_dir + '*.png')):
            face.append(img)
    assert len(face) == 29416, "The total number of images here is not correct."
    celeb_datasize = len(face)
    validation_count = int(validation_split * celeb_datasize / 100)
    train_count = celeb_datasize - validation_count

    #Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(Size_for_image),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])
    full_dataset = CelebDataset(face, transform=transform)
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

def get_sfsnet_dataset(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0,
                       training_syn=False):
    albedo = []
    sh = []
    mask = []
    normal = []
    face = []
    depth = []

    if training_syn:
        read_celeba_csv = None
    if read_from_csv is None:
        for img in sorted(glob.glob(syn_dir + '*/*_albedo_*')):
            albedo.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_face_*')):
            face.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_normal_*')):
            normal.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_depth_*')):
            depth.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_mask_*')):
            mask.append(img)

        for img in sorted(glob.glob(syn_dir + '*/*_light_*.txt')):
            sh.append(img)
    else:
        df = pd.read_csv(read_from_csv)
        df = df[:read_first]
        albedo = list(df['albedo'])
        face = list(df['face'])
        normal = list(df['normal'])
        depth = list(df['depth'])
        mask = list(df['mask'])
        sh = list(df['light'])

        name_to_list = {'albedo': albedo, 'normal': normal, 'depth': depth, \
                        'mask': mask, 'face': face, 'light': sh}

        for _, v in name_to_list.items():
            v[:] = [syn_dir + el for el in v]

        # Merge Synthesized Celeba dataset for Psedo-Supervised training
        if read_celeba_csv is not None:
            df = pd.read_csv(read_celeba_csv)
            df = df[:read_first]
            albedo += list(df['albedo'])
            face += list(df['face'])
            normal += list(df['normal'])
            depth += list(df['depth'])
            mask += list(df['mask'])
            sh += list(df['light'])

    assert (len(albedo) == len(face) == len(normal) == len(depth) == len(mask) == len(sh))
    dataset_size = len(albedo)
    validation_count = int(validation_split * dataset_size / 100)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(Size_for_image),
        transforms.ToTensor(),
    ])

    full_dataset = SfSNetDataset(albedo, face, normal, mask, sh, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class CelebDataset(Dataset):
    def __init__(self, face=None, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.mask_tranform = transforms.Compose([
                             transforms.Resize(Size_for_image),
                             transforms.ToTensor(),
                           ])
        self.normal_transform = transforms.Compose([
                                transforms.Resize(Size_for_image),
                                transforms.ToTensor(),
                           ])

    def __getitem__(self, index):
        face = self.transform(Image.open(self.face[index]))
        return face

    def __len__(self):
        return self.dataset_len

class SfSNetDataset(Dataset):
    def __init__(self, albedo, face, normal, mask, sh, transform = None):
        self.albedo = albedo
        self.face   = face
        self.normal = normal
        self.mask   = mask
        self.sh     = sh
        self.transform = transform
        self.dataset_len = len(self.albedo)
        self.mask_transform = transforms.Compose([
                              transforms.Resize(Size_for_image),
                              transforms.ToTensor(),
                            ])
        self.normal_transform = transforms.Compose([
                              transforms.Resize(Size_for_image),
                            ])

    def __getitem__(self, index):
        albedo = self.transform(Image.open(self.albedo[index]))
        face   = self.transform(Image.open(self.face[index]))
        # normal = io.imread(self.face[index]))
        normal = self.normal_transform(Image.open(self.normal[index]))
        normal = torch.tensor(np.asarray(normal)).permute([2, 0, 1])
        normal = normal.type(torch.float)
        normal = (normal - 128) / 128
        if self.mask[index] == 'None':
            # Load dummy 1 mask for CelebA
            # To ensure consistency if mask is used
            mask = torch.ones(3, Size_for_image, Size_for_image)
        else:
            mask   = self.mask_transform(Image.open(self.mask[index]))
        pd_sh  = pd.read_csv(self.sh[index], sep='\t', header = None)
        sh     = torch.tensor(pd_sh.values).type(torch.float).reshape(-1)
        return albedo, normal, mask, sh, face

    def __len__(self):
        return self.dataset_len




