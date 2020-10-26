import torch
from torchvision import transforms
from loader import get_sfsnet_dataset
from torch.utils.data import DataLoader, Dataset
from utils import predict_celeba, predict_sfsnet
from backbone import SfsNetPipeline, SkipNet
import wandb
from loader import wandb_log_images
import os

def Validate_Skip(model_dir=None, syn_dir=None, out_dir=None):
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_dir + 'train/',
                                                    validation_split=2, training_syn=True)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=True)
    validation_len = len(val_dl)
    sfsnet_model = SkipNet()
    sfsnet_model.load_state_dict(torch.load(model_dir))
    suffix = 'Val'

    wandb.init(tensorboard=True)
    for bix, data in enumerate(val_dl):
        print("This is the {}th bix".format(bix))
        if bix % 5 == 0:
            out_dir_cur = out_dir + str(bix)
            if not os.path.exists(out_dir_cur):
                os.system("mkdir "+ out_dir_cur)
            out_dir_cur += '/'
            albedo, normal, mask, sh, face = data
            predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfsnet_model(face)
            wandb_log_images(wandb, predicted_normal, mask, suffix + ' Predicted Normal', bix,
                             suffix + ' Predicted Normal', path= out_dir_cur + 'predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, mask, suffix + ' Predicted Albedo', bix,
                             suffix + ' Predicted Albedo', path= out_dir_cur + 'predicted_albedo.png')
            wandb_log_images(wandb, out_shading, mask, suffix + ' Predicted Shading', bix,
                             suffix + ' Predicted Shading', path= out_dir_cur + 'predicted_shading.png')
            wandb_log_images(wandb, out_recon, mask, suffix + ' Out recon', bix,
                             suffix + ' Out recon', path= out_dir_cur + 'out_recon.png')
            print("We finished the logging process at the {}th bix".format(bix))

if __name__ == "__main__":
    model_dir = "/home/hd8t/xiangyu.yin/results/metadata/checkpoints/Skip_First.pkl"
    syn_dir = "/home/hd8t/xiangyu.yin/DATA_pose_15/"
    out_dir = "/home/hd8t/xiangyu.yin/results/metadata/out_images/val/"
    Validate_Skip(model_dir, syn_dir, out_dir)