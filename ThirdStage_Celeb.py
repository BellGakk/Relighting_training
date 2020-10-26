import torch
import torch.nn as nn
from loader import get_celeba_dataset, get_sfsnet_dataset
from backbone import SkipNet, SfsNetPipeline
from torch.utils.data import DataLoader, Dataset
import os
from loader import apply_mask, de_norm, wandb_log_images
import wandb

def thirdStageTraining(syn_data, celeb_data, batch_size=8, num_epochs=20, log_path=None,
                       use_cuda=True, lr=0.0025, weight_decay=0.005):

    train_celeb_dataset, val_celeb_dataset = get_celeba_dataset(celeb_dir=celeb_data, validation_split=10)
    test_celeb_dataset, _ = get_celeba_dataset(celeb_dir=celeb_data, validation_split=0)

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_images_dir = log_path + 'out_images/'
    out_celeb_images_dir = out_images_dir + 'celeb/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    prev_SkipNet_model = SkipNet()
    prev_SkipNet_model.load_state_dict(torch.load('/home/hd8t/xiangyu.yin/results/metadata/checkpoints/Skip_First.pkl'))
    prev_SkipNet_model.to(device)

    os.system("mkdir -p {}".format(model_checkpoint_dir))
    os.system("mkdir -p {}".format(out_celeb_images_dir + 'train/'))
    os.system("mkdir -p {}".format(out_celeb_images_dir + 'val/'))
    os.system("mkdir -p {}".format(out_celeb_images_dir + 'test/'))

    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    sh_loss = nn.MSELoss()
    recon_loss = nn.L1Loss()
    c_recon_loss = nn.L1Loss()
    c_sh_loss = nn.MSELoss()
    c_albedo_loss = nn.L1Loss()
    c_normal_loss = nn.L1Loss()

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss = sh_loss.cuda()
        recon_loss = recon_loss.cuda()
        c_recon_loss = c_recon_loss.cuda()
        c_sh_loss = c_sh_loss.cuda()
        c_albedo_loss = c_albedo_loss.cuda()
        c_normal_loss = c_normal_loss.cuda()


    lamda_recon = 0.5
    lamda_normal = 0.5
    lamda_albedo = 0.5
    lamda_sh = 0.1


    wandb.init(tensorboard=True)
    for epoch in range(1, num_epochs+1):
        tloss = 0
        nloss = 0
        aloss = 0
        shloss = 0
        rloss = 0
        c_tloss = 0
        c_nloss = 0
        c_aloss = 0
        c_shloss = 0
        c_reconloss = 0
        predicted_normal = None
        predicted_albedo = None
        out_shading = None
        out_recon = None
        mask = None
        face = None
        normal = None
        albedo = None
        c_predicted_normal = None
        c_predicted_albedo = None
        c_out_shading = None
        c_out_recon = None
        c_face = None

        sfsnet_model = SfsNetPipeline()
        if epoch > 1:
            sfsnet_model.load_state_dict(torch.load('/home/hd8t/xiangyu.yin/results/metadata/checkpoints/SfsNet_Celeb.pkl'))
        model_parameters = sfsnet_model.parameters()
        optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
        sfsnet_model.to(device)

        celeb_train_dl = DataLoader(train_celeb_dataset, batch_size=batch_size, shuffle=True)
        celeb_val_dl = DataLoader(val_celeb_dataset, batch_size=batch_size, shuffle=False)
        celeb_test_dl = DataLoader(test_celeb_dataset, batch_size=batch_size, shuffle=True)

        celeb_train_len = len(celeb_train_dl)

        if epoch == 0:
            print("Celeb dataset: Train data:", len(celeb_train_dl), ' Val data: ', len(celeb_val_dl), ' Test data: ',
                  len(celeb_test_dl))
        celeb_train_iter = iter(celeb_train_dl)
        celeb_count = 0
        #Until we process all synthetic and celebA data
        while True:
            c_data = next(celeb_train_iter, None)
            if c_data is not None:
                celeb_count += 1
                c_mask = None
                if use_cuda:
                    c_data = c_data.cuda()

                c_face = c_data
                prevc_normal, prevc_albedo, prevc_sh, prevc_shading, prec_recon = prev_SkipNet_model(c_face)
                c_predicted_normal, c_predicted_albedo, c_predicted_sh, c_out_shading, c_out_recon = sfsnet_model(c_face)

                c_current_normal_loss = c_normal_loss(c_predicted_normal, prevc_normal)
                c_current_albedo_loss = c_albedo_loss(c_predicted_albedo, prevc_albedo)
                c_current_sh_loss = c_sh_loss(c_predicted_sh, prevc_sh)
                c_current_recon_loss = c_recon_loss(c_out_recon, de_norm(c_face))

                c_total_loss = lamda_sh * c_current_sh_loss + lamda_normal * c_current_normal_loss + lamda_albedo * c_current_albedo_loss +\
                              lamda_recon * c_current_recon_loss

                optimizer.zero_grad()
                c_total_loss.backward()
                optimizer.step()

                c_tloss += c_total_loss.item()
                c_nloss += c_current_normal_loss.item()
                c_aloss += c_current_albedo_loss.item()
                c_shloss += c_current_sh_loss.item()
                c_reconloss += c_current_recon_loss.item()
                print("Epoch {}/20, celeb data {}/{}. Celeb total loss: {}, normal_loss: {}, albedo_loss: {}, sh_loss: {}, recon_loss: {}".format(
                    epoch, celeb_count, celeb_train_len, c_total_loss, c_current_normal_loss, c_current_albedo_loss, c_current_sh_loss, c_current_recon_loss
                ))
            elif c_data is None:
                break

            # Log CelebA image
        file_name = out_celeb_images_dir + 'train/' + 'train_' + str(epoch)
        wandb_log_images(wandb, c_predicted_normal, None, 'Train CelebA Predicted Normal', epoch,
                            'Train CelebA Predicted Normal', path=file_name + '_c_predicted_normal.png')
        wandb_log_images(wandb, c_predicted_albedo, None, 'Train CelebA Predicted Albedo', epoch,
                            'Train CelebA Predicted Albedo', path=file_name + '_c_predicted_albedo.png')
        wandb_log_images(wandb, c_out_shading, None, 'Train CelebA Predicted Shading', epoch,
                            'Train CelebA Predicted Shading', path=file_name + '_c_predicted_shading.png',
                            denormalize=False)
        wandb_log_images(wandb, c_out_recon, None, 'Train CelebA Recon', epoch, 'Train CelebA Recon',
                            path=file_name + '_c_predicted_face.png', denormalize=False)
        wandb_log_images(wandb, c_face, None, 'Train CelebA Ground Truth', epoch, 'Train CelebA Ground Truth',
                            path=file_name + '_c_gt_face.png')
        torch.save(sfsnet_model.state_dict(), "/home/hd8t/xiangyu.yin/results/metadata/checkpoints/SfsNet_Celeb.pkl")




if __name__ == "__main__":
    syn_data = "/home/hd8t/xiangyu.yin/DATA_pose_15/"
    celeb_data = '/home/hd8t/data/CelebA-HQ/original/'
    log_path = '/home/hd8t/xiangyu.yin/results/metadata/'
    thirdStageTraining(syn_data=syn_data, celeb_data=celeb_data, log_path=log_path)