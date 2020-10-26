from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from loader import get_sfsnet_dataset, get_celeba_dataset, CelebDataset
from backbone import SfsNetPipeline, SkipNet
from utils import wandb_log_images, random_split
import torch
import os
import glob
import wandb

#def predict_celeb(celeb_path=None, out_dir=None):
#     return True

if __name__ == "__main__":
    Size_for_Image = 128
    suffix = 'celeba'
    check_dir = "/home/hd8t/xiangyu.yin/results/metadata/checkpoints/Skip_First.pkl"
    Celeb_path = "/home/hd8t/data/CelebA-HQ/original/"
    out_dir = "/home/hd8t/xiangyu.yin/results/metadata/out_images/celeba/"
    sfsnet_model = SkipNet()
    sfsnet_model.load_state_dict(torch.load("/home/hd8t/xiangyu.yin/results/metadata/checkpoints/Skip_First.pkl"))
    face = []
    name = []
    for img in glob.glob(Celeb_path + "*.png"):
        n_suffix = img.split('/')[-1]
        face.append(img)
        name.append(n_suffix.split('.')[0])
    datasize = len(face)
    validation_count = int(2 * datasize / 100)
    train_count = datasize - validation_count
    transform = transforms.Compose([
        transforms.Resize(Size_for_Image),
        transforms.ToTensor()
    ])
    full_dataset = CelebDataset(face, name, transform)
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    celeb_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    wandb.init(tensorboard=True)
    for bix, data in enumerate(celeb_dl):
        fa, na = data
        na = na[0]
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, out_recon = sfsnet_model(fa)
        print('Consecute the {}th face'.format(bix))
        if bix % 10 == 0:
            print(predicted_albedo)
            print(predicted_shading)
            out_celeb = out_dir + 'Celeb' + str(bix)
            if not os.path.exists(out_celeb):
                os.system("mkdir " + out_celeb)
            out_celeb += "/"
            wandb_log_images(wandb, predicted_normal, None, suffix + "Predicted Normal", \
                             bix, suffix + "Predicted Normal", path=out_celeb + na + "_predicted_normal.png")
            wandb_log_images(wandb, predicted_albedo, None, suffix + "Predicted Albedo", \
                             bix, suffix + "Predicted Albedo", path=out_celeb + na + "_predicted_albedo.png")
            wandb_log_images(wandb, predicted_shading, None, suffix + "Predicted Shading", \
                             bix, suffix + "Predicted Shading", path=out_celeb + na + "_predicted_shading.png")
            wandb_log_images(wandb, out_recon, None, suffix + 'Out Recon', \
                             bix, suffix + 'Out Recon', path=out_celeb + na + "_out_recon.png")
