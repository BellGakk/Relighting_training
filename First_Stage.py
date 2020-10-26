import torch
import torch.nn as nn
from backbone import SkipNet
from torchvision import transforms
import numpy
from loader import get_sfsnet_dataset
from torch.utils.data import DataLoader, Dataset

def FirstStage_Training(syn_path=None, model_dir=None):

    learning_rate = 0.00125
    weight_decay = 0.0001
    if torch.cuda.is_available():
        use_cuda = True
    nums_epoch = 20
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_path + 'train/',
                                                    validation_split=2, training_syn=True)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_path + 'test/', validation_split=0,
                                         training_syn=True)

    normal_loss = nn.L1Loss()
    albedo_loss = nn.L1Loss()
    lighting_loss = nn.MSELoss()
    recon_loss = nn.L1Loss()

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        lighting_loss = lighting_loss.cuda()
        recon_loss = recon_loss.cuda()

    lambda_normal = 0.5
    lambda_albedo = 0.5
    lambda_sh = 0.1
    lambda_recon = 0.5

    # wandb.init(tensorboard=True)
    for epoch in range(nums_epoch):

        syn_train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
        syn_val_dl = DataLoader(val_dataset, batch_size=32, shuffle=True)
        syn_test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
        print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ',
              len(syn_test_dl))
        len_syn_train = len(syn_train_dl)
        t_loss = 0
        n_loss = 0
        a_loss = 0
        sh_loss = 0
        r_loss = 0
        sfsnet_model = SkipNet()
        if epoch > 0:
            sfsnet_model.load_state_dict(torch.load(model_dir + "Skip_First" + ".pkl"))
        sfsnet_model.to(device)
        parameters = sfsnet_model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

        for bix, data in enumerate(syn_train_dl):
            albedo, normal, mask, sh, face = data
            print(albedo.shape)
            print(normal.shape)
            print(mask.shape)
            print(face.shape)
            print(sh.shape)
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask = mask.cuda()
                sh = sh.cuda()
                face = face.cuda()
            print('True')

            predicted_normal, predicted_albedo, predicted_sh, produced_shading, produced_recon = sfsnet_model(face)
            current_normal_loss = normal_loss(predicted_normal, normal)
            current_albedo_loss = albedo_loss(predicted_albedo, albedo)
            current_sh_loss = lighting_loss(predicted_sh, sh)
            current_recon_loss = recon_loss(produced_recon, face)

            total_loss = lambda_normal * current_normal_loss + lambda_albedo * current_albedo_loss + \
                         lambda_sh * current_sh_loss + lambda_recon * current_recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            t_loss += total_loss.item()
            a_loss += current_albedo_loss.item()
            n_loss += current_normal_loss.item()
            sh_loss += current_sh_loss.item()
            r_loss += current_recon_loss.item()

            print('Epoch: {} - Total Loss : {}, Normal Loss: {}, Albedo Loss: {}, SH Loss:{}, Recon Loss:{}'.format(
                epoch, \
                total_loss, current_albedo_loss, current_normal_loss, current_sh_loss, current_recon_loss))
            print('This is {} / {} of training dataline'.format(bix, (len(syn_train_dl) - 1)))

        torch.save(sfsnet_model.state_dict(), model_dir + "Skip_First" + ".pkl")

if __name__ == "__main__":
    syn_path = "/home/hd8t/xiangyu.yin/DATA_pose_15/"
    model_dir = "/home/hd8t/xiangyu.yin/results/metadata/checkpoints/"
    FirstStage_Training(syn_path, model_dir)