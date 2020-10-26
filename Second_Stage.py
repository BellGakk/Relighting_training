from loader import get_sfsnet_dataset, get_normal_in_range, wandb_log_images
from backbone import SkipNet_Encoder, SkipNet_Decoder, SkipNet, SfsNetPipeline
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
import torch.nn as nn
import wandb

def train_synthetic(sfs_net_model, syn_data, celeba_data=None, read_first=None,
                    batch_size=10, num_epochs=10, log_path='./results/metadata/', use_cuda=False, wandb=None,
                    lr=0.01, wt_decay=0.005, training_syn=False):
    # data processing
    syn_train_csv = syn_data + '/train.csv'
    syn_test_csv = syn_data + '/test.csv'

    celeba_train_csv = None
    celeba_test_csv = None
    val_celeba_dl = None

    # We don't train celeba_data so we don't need these codes here.
    if celeba_data is not None:
        celeba_train_csv = celeba_data + '/train.csv'
        celeba_test_csv = celeba_data + '/test.csv'

    #    if training_syn:
    #        celeba_dt, _ = get_celeba_dataset(read_from_csv=celeba_train_csv, read_first=batch_size, validation_split=0)
    #        val_celeba_dl = DataLoader(celeba_dt, batch_size=batch_size, shuffle=True)

    # Load Synthetic dataset
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_data + 'train/', read_from_csv=syn_train_csv,
                                                    read_celeba_csv=celeba_train_csv, read_first=read_first,
                                                    validation_split=2, training_syn=training_syn)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data + 'test/', read_from_csv=syn_test_csv,
                                         read_celeba_csv=celeba_test_csv, read_first=100, validation_split=0,
                                         training_syn=training_syn)

    syn_train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Synthetic dataset: Train data: ', len(syn_train_dl), ' Val data: ', len(syn_val_dl), ' Test data: ',
          len(syn_test_dl))

    model_checkpoint_dir = log_path + 'checkpoints/'
    out_syn_images_dir = log_path + 'out_images/'

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'val/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + 'test/'))
    if val_celeba_dl is not None:
        os.system('mkdir -p {}'.format(out_syn_images_dir + 'celeba_val/'))

    # Collect model parameters
    model_parameters = sfs_net_model.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=wt_decay)
    normal_loss = nn.MSELoss()
    albedo_loss = nn.MSELoss()
    sh_loss = nn.MSELoss()
    recon_loss = nn.MSELoss()

    if use_cuda:
        normal_loss = normal_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        sh_loss = sh_loss.cuda()
        recon_loss = recon_loss.cuda()

    lamda_recon = 1
    lamda_albedo = 1
    lamda_normal = 1
    lamda_sh = 1

    syn_train_len = len(syn_train_dl)

    for epoch in range(1, num_epochs + 1):
        tloss = 0  # Total loss
        nloss = 0  # Normal loss
        aloss = 0  # Albedo loss
        shloss = 0  # SH loss
        rloss = 0  # Reconstruction loss

        for bix, data in enumerate(syn_train_dl):
            albedo, normal, mask, sh, face = data
            if use_cuda:
                albedo = albedo.cuda()
                normal = normal.cuda()
                mask = mask.cuda()
                sh = sh.cuda()
                face = face.cuda()

            # Apply Mask on input image
            # face = applyMask(face, mask)
            predicted_normal, predicted_albedo, predicted_sh, out_shading, out_recon = sfs_net_model(face)

            # Loss computation
            # Normal loss
            current_normal_loss = normal_loss(predicted_normal, normal)
            # Albedo loss
            current_albedo_loss = albedo_loss(predicted_albedo, albedo)
            # SH loss
            current_sh_loss = sh_loss(predicted_sh, sh)
            # Reconstruction loss
            # Edge case: Shading generation requires denormalized normal and sh
            # Hence, denormalizing face here
            current_recon_loss = recon_loss(out_recon, face)

            total_loss = lamda_normal * current_normal_loss \
                         + lamda_albedo * current_albedo_loss + lamda_sh * current_sh_loss  # + lamda_recon * current_recon_loss

            optimizer.zero_grad()
            total_loss.zero_grad()
            optimizer.step()

            # Logging for display and debugging purposes
            tloss += total_loss.item()
            nloss += current_normal_loss.item()
            aloss += current_albedo_loss.item()
            shloss += current_sh_loss.item()
            rloss += current_recon_loss.item()

        print('Epoch: {} - Total Loss : {}. Normal Loss: {}, Albedo Loss: {}, SH Loss:{}, Recon Loss:{}'.format(epoch,
                                                                                                                tloss, \
                                                                                                                nloss,
                                                                                                                aloss,
                                                                                                                shloss,
                                                                                                                rloss))

        # if we got celeba data, then we are mix dataset.
        log_prefix = 'Syn_Data'
        if celeba_data is not None:
            log_prefix = 'Mix Data'

        # if the number of epoch is odd, then we are in in the synthetic dataset.
        if epoch % 1 == 0:
            print(
                'Training set results: Total Loss: {}, Normal Loss: {}, ALbedo Loss:{}, SH Loss:{}, Recon Loss: {}'.format(
                    tloss / syn_train_len, \
                    nloss / syn_train_len, aloss / syn_train_len, shloss / syn_train_len, rloss / syn_train_len))
            # Log training info
            # wandb.log({log_prefix + 'Train_Total_Loss': tloss/syn_train_len, log_prefix + 'Train_Normal_Loss': nloss/syn_train_len, log_prefix + 'Train_Albedo_Loss':aloss/ syn_train_len,\
            #           log_prefix + 'Train_sh_Loss': shloss/syn_train_len, log_prefix + 'Train_Reconstruction_Loss': rloss/syn_train_len})

            # Log images in wandb
            file_name = out_syn_images_dir + 'train/' + 'train_' + str(epoch)
            save_predicted_normal = get_normal_in_range(predicted_normal)
            save_gt_normal = get_normal_in_range(normal)
            # wandb_log_images(wandb, save_predicted_normal, mask, 'Train Predicted Normal', epoch, 'Train Predicted Normal',
            #                 path=file_name + '_predicted_normal.png')
            # wandb_log_images(wandb, predicted_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo',
            #                 path=file_name + '_predicted_albedo.png')
            # wandb_log_images(wandb, out_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading',
            #                 path=file_name + '_predicted_shading.png', denormalize=False)
            # wandb_log_images(wandb, out_recon, mask, 'Train Recon', epoch, 'Train Recon',
            #                 path=file_name + '_predicted_face.png')
            # wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth',
            #                 path=file_name + '_gt_face.png')
            # wandb_log_images(wandb, save_gt_normal, mask, 'Train Ground Truth Normal', epoch,
            #                 'Train Ground Truth Normal', path=file_name + '_gt_normal.png')
            # wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo',
            #                 path=file_name + '_gt_albedo.png')
            # Get face with real_sh, predicted normal and albedo for debugging
            real_sh_face = sfs_net_model.get_face(sh, predicted_normal, predicted_albedo)
            syn_face = sfs_net_model.get_face(sh, normal, albedo)
            # wandb_log_images(wandb, real_sh_face, mask, 'Train Real SH Predicted Face', epoch,
            #                 'Train Real SH Predicted Face', path=file_name + '_real_sh_face.png')
            # wandb_log_images(wandb, syn_face, mask, 'Train Real SH GT Face', epoch, 'Train Real SH GT Face',
            #                 path=file_name + '_syn_gt_face.png')

            v_total, v_normal, v_albedo, v_sh, v_recon = predict_sfsnet(sfs_net_model, syn_val_dl,
                                                                        train_epoch_num=epoch, use_cuda=use_cuda,
                                                                        out_folder=out_syn_images_dir + '/val/',
                                                                        wandb=None)
            # wandb.log({log_prefix + 'Val Total loss': v_total, log_prefix + 'Val Albedo loss': v_albedo,
            #           log_prefix + 'Val Normal loss': v_normal, \
            #           log_prefix + 'Val SH loss': v_sh, log_prefix + 'Val Recon loss': v_recon})

            print(
                'Val set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}'.format(
                    v_total,
                    v_normal, v_albedo, v_sh, v_recon))

            #            if val_celeba_dl is not None:
            #                predict_celeba(sfs_net_model, val_celeba_dl, train_epoch_num=0,
            #                               use_cuda=use_cuda, out_folder=out_syn_images_dir + 'celeba_val/', wandb=wandb,
            #                               dump_all_images=True)

            # Model saving
            torch.save(sfs_net_model.state_dict(), model_checkpoint_dir + 'skipnet_model.pkl')
        if epoch % 5 == 0:
            t_total, t_normal, t_albedo, t_sh, t_recon = predict_sfsnet(sfs_net_model, syn_test_dl,
                                                                        train_epoch_num=epoch, use_cuda=use_cuda,
                                                                        out_folder=out_syn_images_dir + '/test/',
                                                                        wandb=None, suffix='Test')

            # wandb.log({log_prefix + 'Test Total loss': t_total, log_prefix + 'Test Albedo loss': t_albedo,
            #           log_prefix + 'Test Normal loss': t_normal, \
            #           log_prefix + 'Test SH loss': t_sh, log_prefix + 'Test Recon loss': t_recon})

            print(
                'Test-set results: Total Loss: {}, Normal Loss: {}, Albedo Loss: {}, SH Loss: {}, Recon Loss: {}\n'.format(
                    t_total,
                    t_normal, t_albedo, t_sh, t_recon))


def train_first_stage(net_model, train_path, test_path, batch_size):
    syn_train_dl = DataLoader(train_path, batch_size=10, shuffle=True)
    syn_test_dl = DataLoader(test_path, batch_size=10, shuffle=True)

    parameters = net_model.parameters()


if __name__ == "__main__":
    learning_rate = 0.00125
    weight_decay = 0.0001
    if torch.cuda.is_available():
        use_cuda = True
    nums_epoch = 20
    syn_data = "/home/hd8t/xiangyu.yin/DATA_pose_15/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset = get_sfsnet_dataset(syn_dir=syn_data + 'train/',
                                                    validation_split=2, training_syn=True)
    test_dataset, _ = get_sfsnet_dataset(syn_dir=syn_data + 'test/', validation_split=0,
                                         training_syn=True)

    log_path = '/home/hd8t/xiangyu.yin/results/metadata/'
    model_checkpoint_dir = log_path + 'checkpoints/'
    out_syn_images_dir = log_path + 'out_images/'

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
        sfsnet_model = SfsNetPipeline()
        sfsnet_model.load_state_dict(torch.load(model_checkpoint_dir + "SfsNet_new_three" + ".pkl"))
        # sfsnet_model.fix_weights()
        # sfsnet_model.load_state_dict(torch.load(model_checkpoint_dir + "SfsNet_new_two" + ".pkl"))
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
            # log image info
            # file_name = out_syn_images_dir + "train/" + "train_" + str(epoch)
            # predicted_normal = get_normal_in_range(predicted_normal)
            # gt_normal = get_normal_in_range(normal)

            # if not os.path.exists(file_name):
            #    os.system("mkdir " + file_name)
            # train_dir_path = file_name + "/" + str(bix)
            # if not os.path.exists(train_dir_path):
            #    os.system("mkdir " + train_dir_path)

            log_prefix = 'syn_train'
            # wandb.log({
            #    log_prefix + "Total Loss": total_loss, log_prefix + "Albedo Loss": albedo_loss,
            #               log_prefix + "Normal Loss": normal_loss, log_prefix + "SH Loss": sh_loss,
            #               log_prefix + "recon Loss": recon_loss
            # })
            '''
            wandb.init(sync_tensorboard=True)
            wandb_log_images(wandb, predicted_normal, mask, 'Train Predicted Normal', epoch,
                             'Train Predicted Normal',
                             path= train_dir_path + '/' + 'predicted_normal.png')
            wandb_log_images(wandb, predicted_albedo, mask, 'Train Predicted Albedo', epoch, 'Train Predicted Albedo',
                             path= train_dir_path + '/' + 'predicted_albedo.png')
            wandb_log_images(wandb, produced_shading, mask, 'Train Predicted Shading', epoch, 'Train Predicted Shading',
                             path= train_dir_path + '/' + 'predicted_shading.png', denormalize=False)
            wandb_log_images(wandb, produced_recon, mask, 'Train Recon', epoch, 'Train Recon',
                             path= train_dir_path + '/' + 'predicted_face.png')
            wandb_log_images(wandb, face, mask, 'Train Ground Truth', epoch, 'Train Ground Truth',
                             path= train_dir_path + '/' + 'gt_face.png')
            wandb_log_images(wandb, gt_normal, mask, 'Train Ground Truth Normal', epoch,
                             'Train Ground Truth Normal', path= train_dir_path + '/' + 'gt_normal.png')
            wandb_log_images(wandb, albedo, mask, 'Train Ground Truth Albedo', epoch, 'Train Ground Truth Albedo',
                             path= train_dir_path + '/' + 'gt_albedo.png')

            '''

        train_txt_file_path = "/home/hd8t/xiangyu.yin/results/metadata/out_images/losses/train_loss.txt"
        train_loss_file = open(train_txt_file_path, "w+")
        train_loss_file.writelines(
            "Training set result: Total loss : {}, Normal loss : {}, Albedo Loss : {}, SH Loss : {}, Recon Loss:{}".format(
                t_loss / len_syn_train, n_loss / len_syn_train, a_loss / len_syn_train, sh_loss / len_syn_train,
                r_loss / len_syn_train))
        print(
            "Training set result: Total loss : {}, Normal loss : {}, Albedo Loss : {}, SH Loss : {}, Recon Loss:{}".format(
                t_loss / len_syn_train, n_loss / len_syn_train, a_loss / len_syn_train, sh_loss / len_syn_train,
                r_loss / len_syn_train))

        torch.save(sfsnet_model.state_dict(), model_checkpoint_dir + "SfsNet_new_three" + ".pkl")

        # wandb.init(sync_tensorboard=True)
        # log training info
        # wandb.log(
        #    {log_prefix + "Total_loss": t_loss / len_syn_train, log_prefix + "Normal_loss": n_loss / len_syn_train, \
        #     log_prefix + "Albedo_loss": a_loss / len_syn_train, log_prefix + "lighting_loss": sh_loss / len_syn_train, \
        #     log_prefix + "recon_loss": r_loss / len_syn_train})

        # val_dir_path = out_syn_images_dir + "val/" + "val_" + str(epoch)
        # if not os.path.exists(val_dir_path):
        #    os.system("mkdir " + val_dir_path)
        # val_dir_path = val_dir_path + "/"
        '''
        vali_total, vali_normal, vali_albedo, vali_sh, vali_recon = predict_sfsnet(sfsnet_model, syn_val_dl,
                                                                                   train_epoch_num=epoch,
                                                                                   use_cuda=use_cuda,
                                                                                   out_folder=val_dir_path,
                                                                                   wandb=wandb, suffix="Val")

        #wandb.log({log_prefix + "vali_Total Loss": vali_total, log_prefix + "vali Albedo Loss": vali_albedo,
        #           log_prefix + "vali Normal Loss": vali_normal, log_prefix + "vali SH Loss": vali_sh,
        #           log_prefix + "vali recon Loss": vali_recon})

        val_txt_file_path = "/home/hd8t/xiangyu.yin/results/metadata/out_images/losses/val_loss.txt"
        val_loss_file = open(val_txt_file_path, "w+")
        val_loss_file.writelines("Val set result: total loss:{}, albedo loss:{}, normal loss:{}, sh_loss:{}, recon_loss:{}\n".format(
            vali_total, vali_albedo, vali_normal, vali_sh, vali_recon))
        print("Val set result: total loss:{}, albedo loss:{}, normal loss:{}, sh_loss:{}, recon_loss:{}".format(
            vali_total, vali_albedo, vali_normal, vali_sh, vali_recon))
        '''
        # Model saving

        '''
        if epoch % 5 == 0:

            wandb.init(tensorboard=True)
            test_dir_path = out_syn_images_dir + "test/" + "test_" + str(epoch)
            if not os.path.exists(test_dir_path):
                os.system("mkdir " + test_dir_path)
            test_dir_path = test_dir_path + "/"

            test_total, test_normal, test_albedo, test_sh, test_recon = predict_sfsnet(sfsnet_model, syn_test_dl, 
                                                                        train_epoch_num=epoch, use_cuda=use_cuda,
                                                                        out_folder= test_dir_path,
                                                                        wandb=wandb, suffix="Test")

            #wandb.log({log_prefix + "Test Total Loss": test_total, log_prefix + "Test Albedo Loss": test_albedo,
            #           log_prefix + "Test Normal Loss": test_normal, log_prefix + "Test SH Loss": test_sh,
            #           log_prefix + "Test recon Loss": test_recon})

            test_txt_file_path = "/home/hd8t/xiangyu.yin/results/metadata/out_images/losses/test_loss.txt"
            test_loss_file = open(test_txt_file_path, "w+")
            test_loss_file.writelines("Test set result: total loss:{}, albedo_loss:{}, normal_loss:{}, sh_loss:{}, recon_loss:{}\n".format(
                test_total, test_albedo, test_normal, test_sh, test_recon))
            print("Test set result: total loss:{}, albedo_loss:{}, normal_loss:{}, sh_loss:{}, recon_loss:{}".format(
                test_total, test_albedo, test_normal, test_sh, test_recon))
        '''

