# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import random
import torch.nn as nn
import scipy.io as scio

from data_loader import VideoDataset
from utils import performance_fit
from utils import plcc_loss

from models import VQAModels

from torchvision import transforms
import time


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main(config):

    all_test_SRCC, all_test_KRCC, all_test_PLCC, all_test_RMSE = [], [], [], []

    for i in range(config.n_exp):
        config.exp_version = i
        print('%d round training starts here' % i)
        seed = i * 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print('The current model is ' + config.model_name)
        if config.model_name == 'Model_I':
            model = VQAModels.Model_I()
        elif config.model_name == 'Model_II':
            model = VQAModels.Model_II()
        elif config.model_name == 'Model_III':
            model = VQAModels.Model_III()
        elif config.model_name == 'Model_IV':
            model = VQAModels.Model_IV()
        elif config.model_name == 'Model_V':
            model = VQAModels.Model_V()
        elif config.model_name == 'Model_VI':
            model = VQAModels.Model_VI()
        elif config.model_name == 'Model_VII':
            model = VQAModels.Model_VII()
        elif config.model_name == 'Model_VIII':
            model = VQAModels.Model_VIII()
        elif config.model_name == 'Model_IX':
            model = VQAModels.Model_IX()
        elif config.model_name == 'Model_X':
            model = VQAModels.Model_X()
        elif config.model_name == 'Model_XI':
            model = VQAModels.Model_XI()

        if config.model_name in ['Model_II', 'Model_V', 'Model_VIII', 'Model_X']:
            print('using the pretrained model')
            model.load_state_dict(torch.load(config.pretrained_path))

        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            model = model.to(device)
        else:
            model = model.to(device)
        

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr = config.lr, weight_decay = 0.0000001)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

        criterion = plcc_loss

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))


        transformations_train = transforms.Compose([transforms.Resize(config.resize), \
                                                    transforms.RandomCrop(config.crop_size), \
                                                    transforms.ToTensor(), \
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.Resize(config.resize), \
                                                   transforms.CenterCrop(config.crop_size), \
                                                   transforms.ToTensor(), \
                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        ## training data
        if config.dataset == 'LSVQ':

            datainfo_train = 'data/LSVQ_whole_train.csv'
            datainfo_val = 'data/LSVQ_whole_train.csv'
            datainfo_test = 'data/LSVQ_whole_test.csv'
            datainfo_test_1080p = 'data/LSVQ_whole_test_1080p.csv'

            trainset = VideoDataset(config.videos_dir, config.feature_dir, datainfo_train, transformations_train, config.dataset+'_train', config.crop_size, seed)
            valset = VideoDataset(config.videos_dir, config.feature_dir, datainfo_val, transformations_train, config.dataset+'_val', config.crop_size, seed)
            testset = VideoDataset(config.videos_dir, config.feature_dir, datainfo_test, transformations_test, config.dataset+'_test', config.crop_size, seed)
            testset_1080p = VideoDataset(config.videos_dir, config.feature_dir, datainfo_test_1080p, transformations_test, config.dataset+'_test_1080p', config.crop_size, seed)
        
        else:

            trainset = VideoDataset(config.videos_dir, config.feature_dir, config.datainfo, transformations_train, config.dataset+'_train', config.crop_size, seed)
            valset = VideoDataset(config.videos_dir, config.feature_dir, config.datainfo, transformations_train, config.dataset+'_val', config.crop_size, seed)
            testset = VideoDataset(config.videos_dir, config.feature_dir, config.datainfo, transformations_test, config.dataset+'_test', config.crop_size, seed)


        
        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, \
                                                   batch_size=config.train_batch_size, \
                                                   shuffle=True, \
                                                   num_workers=config.num_workers)

        val_loader = torch.utils.data.DataLoader(valset, \
                                                 batch_size=1, \
                                                 shuffle=True, \
                                                 num_workers=config.num_workers)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
            shuffle=False, num_workers=config.num_workers)

        if config.dataset == 'LSVQ':
            test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, \
                                                            batch_size=1, \
                                                            shuffle=False, \
                                                            num_workers=config.num_workers)


        best_test_criterion = -1  # SROCC min
        best_test = []
        if config.dataset == 'LSVQ':
            best_test_1080p = []

        print('Starting training:')

        old_save_name = None
        old_mat_name = None

        for epoch in range(config.epochs):
            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (video, temporal_features, mos, _) in enumerate(train_loader):

                video = video.to(device)
                temporal_features = temporal_features.to(device)
                labels = mos.to(device).float()
                
                if config.model_name in ['Model_IV', 'Model_V', 'Model_VI', 'Model_IX', 'Model_X', 'Model_XI']:
                    outputs = model(video, temporal_features)
                else:
                    outputs = model(video)
                optimizer.zero_grad()
                
                loss = criterion(labels, outputs)
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                loss.backward()
                
                optimizer.step()

                if (i+1) % (config.print_samples//config.train_batch_size) == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                        (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                            avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))



            # Val
            with torch.no_grad():
                model.eval()
                label = np.zeros([len(valset)])
                y_output = np.zeros([len(valset)])
                for i, (video, temporal_features, mos, _) in enumerate(val_loader):
                    
                    video = video.to(device)
                    temporal_features = temporal_features.to(device)
                    label[i] = mos.item()
                    if config.model_name in ['Model_IV', 'Model_V', 'Model_VI', 'Model_IX', 'Model_X', 'Model_XI']:
                        outputs = model(video, temporal_features)
                    else:
                        outputs = model(video)

                    y_output[i] = outputs.item()
                
                val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
                
                print('Epoch {} completed. The result on the val dataset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                    val_SRCC, val_KRCC, val_PLCC, val_RMSE))

                # Test
                label = np.zeros([len(testset)])
                y_output = np.zeros([len(testset)])
                for i, (video, temporal_features, mos, _) in enumerate(test_loader):
                    
                    video = video.to(device)
                    temporal_features = temporal_features.to(device)
                    label[i] = mos.item()
                    if config.model_name in ['Model_IV', 'Model_V', 'Model_VI', 'Model_IX', 'Model_X', 'Model_XI']:
                        outputs = model(video, temporal_features)
                    else:
                        outputs = model(video)

                    y_output[i] = outputs.item()
                
                test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)
                
                print('Epoch {} completed. The result on the test dataset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                    test_SRCC, test_KRCC, test_PLCC, test_RMSE))
                
                
                if config.dataset == 'LSVQ':
                    label_1080p = np.zeros([len(testset_1080p)])
                    y_output_1080p = np.zeros([len(testset_1080p)])
                    for i, (video, temporal_features, mos, _) in enumerate(test_loader_1080p):
                        
                        video = video.to(device)
                        temporal_features = temporal_features.to(device)
                        label_1080p[i] = mos.item()
                        if config.model_name in ['Model_IV', 'Model_V', 'Model_VI', 'Model_IX', 'Model_X', 'Model_XI']:
                            outputs = model(video, temporal_features)
                        else:
                            outputs = model(video)

                        y_output_1080p[i] = outputs.item()
                    
                    test_PLCC_1080p, test_SRCC_1080p, test_KRCC_1080p, test_RMSE_1080p = performance_fit(label_1080p, y_output_1080p)
                    
                    print('Epoch {} completed. The result on the test_1080p dataset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(epoch + 1, \
                        test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p))
                    
                if val_SRCC > best_test_criterion:
                    print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                    best_test_criterion = val_SRCC
                    best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                    if config.dataset == 'LSVQ':
                        best_test_1080p = [test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p]
                    print('Saving model...')
                    if not os.path.exists(config.ckpt_path):
                        os.makedirs(config.ckpt_path)

                    if epoch > 0:
                        if os.path.exists(old_save_name):
                            os.remove(old_save_name)
                        if os.path.exists(old_mat_name):
                            os.remove(old_mat_name)

                    save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                        config.dataset + '_NR_v'+ str(config.exp_version) \
                            + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))

                    save_mat_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                        config.dataset + '_NR_v'+ str(config.exp_version) \
                            + '_epoch_%d_SRCC_%f.mat' % (epoch + 1, test_SRCC))
                    if config.dataset == 'LSVQ':
                        scio.savemat(save_mat_name, {'y_output':y_output, 'label':label, 'y_output_1080p':y_output_1080p, 'label_1080p':label_1080p})
                    else:
                        scio.savemat(save_mat_name, {'y_output':y_output, 'label':label})
                    torch.save(model.module.state_dict(), save_model_name)
                    old_save_name = save_model_name
                    old_mat_name = save_mat_name


            print('Training completed.')
            print('The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                best_test[0], best_test[1], best_test[2], best_test[3]))
            if config.dataset == 'LSVQ':
                print('The best training result on the test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
                    best_test_1080p[0], best_test_1080p[1], best_test_1080p[2], best_test_1080p[3]))
            print('*************************************************************************************************************************')

        all_test_SRCC.append(best_test[0])
        all_test_KRCC.append(best_test[1])
        all_test_PLCC.append(best_test[2])
        all_test_RMSE.append(best_test[3])


    print('*************************************************************************************************************************')
    print('SRCC:')
    print(all_test_SRCC)
    print('KRCC:')
    print(all_test_KRCC)
    print('PLCC:')
    print(all_test_PLCC)
    print('RMSE:')
    print(all_test_RMSE)
    print(
        'The avg results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.mean(all_test_SRCC), np.mean(all_test_KRCC), np.mean(all_test_PLCC), np.mean(all_test_RMSE)))

    print(
        'The std results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.std(all_test_SRCC), np.std(all_test_KRCC), np.std(all_test_PLCC), np.std(all_test_RMSE)))

    print(
        'The median results SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            np.median(all_test_SRCC), np.median(all_test_KRCC), np.median(all_test_PLCC), np.median(all_test_RMSE)))
    print('*************************************************************************************************************************')
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--datainfo', type=str, default=None)
    parser.add_argument('--videos_dir', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--print_samples', type=int, default = 0)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--n_exp', type=int)
    parser.add_argument('--sample_rate', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=0)

    
    config = parser.parse_args()

    torch.manual_seed(config.random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    main(config)













