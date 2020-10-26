from loader import get_sfsnet_dataset, get_normal_in_range, wandb_log_images
from backbone import SkipNet_Encoder, SkipNet_Decoder, SkipNet, SfsNetPipeline
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
import torch.nn as nn
import wandb

def predict_celeba(sfsnet_model, dl, use_cuda=False, out_folder=None, wandb=None, suffix='celeba'):
    tloss = 0
    recon_loss = nn.L1Loss()
    if use_cuda:
        recon_loss = recon_loss.cuda()

    for bix, data in enumerate(dl):
        face = data
        if use_cuda:
            face = face.cuda()
        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfsnet_model(face)
        print("we have computed the No.{} face decomposition.".format(bix))
        if bix % 10 == 0:
            if not os.path.exists(out_folder + str(bix)):
                os.system("mkdir " + out_folder + str(bix))
            file_name = out_folder + str(bix) + '/'
            predicted_normal = get_normal_in_range(predicted_normal)
            wandb_log_images(wandb, predicted_normal, None, suffix + 'Predicted Normal', bix,
                             suffix + ' Predicted Normal', path=file_name + '_predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, None, suffix + 'Predicted Albedo', bix,
                             suffix + 'Predicted Albedo', path=file_name + '_predicted_albedo.png')
            wandb_log_images(wandb, predicted_shading, None, suffix + 'Predicted Shading', bix,
                             suffix + 'Predicted_shading', path=file_name + '_predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, predicted_face, None, suffix + 'Predicted face',
                             path=file_name + '_predicted_face.png', denormalize=False)
            wandb_log_images(wandb, face, None, suffix + ' Ground Truth', bix, suffix + ' Ground Truth',
                             path=file_name + '_gt_face.png')

            total_loss = recon_loss(predicted_face, face)
            tloss += total_loss.item()

    print("tloss is equal to", tloss)


def predict_sfsnet(sfs_net_model, dl, train_epoch_num=0, use_cuda=False, out_folder=None, wandb=None, suffix='Val'):
    # debugging flag to dump image

    fix_bix_dump = 0
    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss = nn.MSELoss()
    recon_loss = nn.L1Loss()

    lamda_recon = 0.5
    lamda_albedo = 0.5
    lamda_normal = 0.5
    lamda_sh = 0.1

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss = sh_loss.cuda()
        recon_loss = recon_loss.cuda()

    tloss = 0  # total loss
    nloss = 0  # normal loss
    aloss = 0  # albedo loss
    shloss = 0  # SH loss
    rloss = 0  # Reconstruction loss

    for bix, data in enumerate(dl):
        albedo, normal, mask, sh, face = data
        if use_cuda:
            albedo = albedo.cuda()
            normal = normal.cuda()
            mask = mask.cuda()
            sh = sh.cuda()
            face = face.cuda()

        predicted_normal, predicted_albedo, predicted_sh, predicted_shading, predicted_face = sfs_net_model(face)

        # save predictions in log folder
        if not os.path.exists(out_folder + str(bix)):
            os.system("mkdir " + out_folder + str(bix))
        file_name = out_folder + str(bix) + "/"
        # log_images
        predicted_normal = get_normal_in_range(predicted_normal)
        gt_normal = get_normal_in_range(normal)

        wandb_log_images(wandb, predicted_normal, mask, suffix + ' Predicted Normal', train_epoch_num,
                         suffix + ' Predicted Normal', path=file_name + 'predicted_normal.png')
        wandb_log_images(wandb, predicted_albedo, mask, suffix + ' Predicted Albedo', train_epoch_num,
                         suffix + ' Predicted Albedo', path=file_name + 'predicted_albedo.png')
        wandb_log_images(wandb, predicted_shading, mask, suffix + ' Predicted Shading', train_epoch_num,
                         suffix + ' Predicted Shading', path=file_name + 'predicted_shading.png',
                         denormalize=False)
        wandb_log_images(wandb, predicted_face, mask, suffix + ' Predicted face', train_epoch_num,
                         suffix + ' Predicted face', path=file_name + 'predicted_face.png', denormalize=False)
        wandb_log_images(wandb, face, mask, suffix + ' Ground Truth', train_epoch_num, suffix + ' Ground Truth',
                         path=file_name + '_gt_face.png')
        wandb_log_images(wandb, gt_normal, mask, suffix + ' Ground Truth Normal', train_epoch_num,
                         suffix + ' Ground Normal', path=file_name + 'gt_normal.png')
        wandb_log_images(wandb, albedo, mask, suffix + ' Ground Truth Albedo', train_epoch_num,
                         suffix + ' Ground Albedo', path=file_name + 'gt_albedo.png')
        # Get face with real SH
        real_sh_face = sfs_net_model.get_face(sh, predicted_normal, predicted_albedo)
        wandb_log_images(wandb, real_sh_face, mask, 'Val Real SH Predicted Face', train_epoch_num,
                         'Val Real SH Predicted Face', path=file_name + 'real_sh_face.png')
        syn_face = sfs_net_model.get_face(sh, normal, albedo)
        wandb_log_images(wandb, syn_face, mask, 'Val Real SH GT Face', train_epoch_num, 'Val Real SH GT Face',
                         path=file_name + '_yn_gt_face.png')

        # TODO
        # Dump SH as CSV or TXT file

        # Loss computation
        # Normal loss
        current_normal_loss = normal_loss(predicted_normal, normal)
        # Albedo loss
        current_albedo_loss = albedo_loss(predicted_albedo, albedo)
        # SH loss
        current_sh_loss = sh_loss(predicted_sh, sh)
        # Reconstruction loss
        current_recon_loss = recon_loss(predicted_face, face)

        total_loss = lamda_recon * current_recon_loss + lamda_normal * current_normal_loss \
                     + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss

        # Logging for display and debugging purposes
        tloss += total_loss.item()
        nloss += current_normal_loss.item()
        aloss += current_albedo_loss.item()
        shloss += current_sh_loss.item()
        rloss += current_recon_loss.item()

    len_dl = len(dl)
    # wandb.log(
    #    {suffix + ' Total loss': tloss / len_dl, 'Val Albedo loss': aloss / len_dl, 'Val Normal loss': nloss / len_dl, \
    #     'Val SH loss': shloss / len_dl, 'Val Recon loss': rloss / len_dl}, step=train_epoch_num)

    # return average loss over dataset
    return tloss / len_dl, nloss / len_dl, aloss / len_dl, shloss / len_dl, rloss / len_dl