import argparse
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
from torchvision import transforms
from torch.utils import data
import scipy.io as scio
import math

import sys


from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list

class SlowFast_feature(torch.nn.Module):
    def __init__(self):
        super(SlowFast_feature, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        # Initialize feature extraction, slow, fast, and adaptive average pooling
        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            
        return slow_feature, fast_feature

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SlowFast_feature()

    model = model.to(device)

    resize = config.resize
    sample_rate = config.sample_rate
    sample_type = config.sample_type
    videos_dir = config.videos_dir
    dataset_file = config.dataset_file
    dataset = config.dataset
    
    transformations_test = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])

    testset = VideoDataset(videos_dir, dataset_file, transformations_test, resize, dataset, sample_rate, sample_type)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    # do validation after each epoch
    with torch.no_grad():
        model.eval()

        for i, (video, video_name) in enumerate(test_loader):
            video_name = video_name[0]
            print(video_name)
            if not os.path.exists(os.path.join(config.feature_save_folder, video_name)):
                os.makedirs(os.path.join(config.feature_save_folder, video_name))
            
            for idx, ele in enumerate(video):
                ele = ele.permute(0, 2, 1, 3, 4)             
                inputs = pack_pathway_output(ele, device)

                slow_feature, fast_feature = model(inputs)
                np.save(os.path.join(config.feature_save_folder, video_name, 'feature_' + str(idx) + '_slow_feature'), slow_feature.to('cpu').numpy())
                np.save(os.path.join(config.feature_save_folder, video_name, 'feature_' + str(idx) + '_fast_feature'), fast_feature.to('cpu').numpy())

class VideoDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize, dataset_name, sample_rate, sample_type):
        super(VideoDataset, self).__init__()

        self.video_names = self.load_video_names(filename_path, dataset_name)
        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.sample_type = sample_type
        self.length = len(self.video_names)

    def load_video_names(self, filename_path, dataset_name):
        """
        Load the names of videos in a dataset from a specified file.

        Args:
            filename_path (str): The path of the file containing the video names.
            dataset_name (str): The name of the dataset. Options include 'KoNViD1k', 'youtube_ugc', 'LIVEVQC', 'LIVE_Qualcomm', 'LBVD', and 'LIVEYTGaming'.

        Returns:
            list: A list of video names in the dataset.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        if dataset_name == 'KoNViD1k' or dataset_name == 'youtube_ugc' or dataset_name == 'LIVEVQC':
            dataInfo = scio.loadmat(filename_path)
            if dataset_name == 'KoNViD1k' or dataset_name == 'youtube_ugc':
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
            else:  # LIVEVQC
                video_names = [dataInfo['video_list'][i][0][0] for i in range(len(dataInfo['video_list']))]
        elif dataset_name == 'LIVE_Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = [video_name.replace('yuv', 'mp4') for video_name in dataInfo['file_names'].tolist()]
        elif dataset_name == 'LBVD':
            dataInfo = scio.loadmat(filename_path)
            video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
        elif dataset_name == 'LIVEYTGaming':
            dataInfo = scio.loadmat(filename_path)
            video_names = [dataInfo['video_list'][i][0][0] + '.mp4' for i in range(len(dataInfo['video_list']))]
        elif 'LSVQ' in dataset_name:
            dataInfo = pd.read_csv(filename_path)
            video_names = [dataInfo['name'].iloc[i] + '.mp4' for i in range(len(dataInfo['name']))]
        else:
            raise ValueError(f"Unsupported database name: {dataset_name}")

        return video_names
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        filename=os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))

        if self.dataset_name == 'LBVD':
            video_frame_rate = int(video_length/10)

        # one clip for one second video
        if video_frame_rate == 0:
            n_clip = 10
        else:
            n_clip = int(video_length/video_frame_rate)

        # Define minimum number of clips based on the dataset
        if self.dataset_name == 'KoNViD1k' or 'LSVQ' in self.dataset_name:
            n_clip_min = 8
        elif self.dataset_name in ['LIVEVQC', 'LIVEYTGaming']:
            n_clip_min = 10
        elif self.dataset_name == 'LBVD':
            n_clip_min = 10
            n_clip = 10
        elif self.dataset_name == 'LIVE_Qualcomm':
            n_clip_min = 15
        elif self.dataset_name == 'youtube_ugc':
            n_clip_min = 20

        # Define the number of frames to sample from each clip
        n_frame_sample = 32             

        # extract all frames of the video
        video_length_all = n_clip * video_frame_rate
        transformed_frame_all = torch.zeros([video_length_all, video_channel, self.resize, self.resize])
        
        video_read_index = 0
        for i in range(video_length_all):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length_all:
            for i in range(video_read_index, video_length_all):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()

        # one chunk for a sample rate second video
        n_chunk = int(math.ceil(n_clip/self.sample_rate))
        n_chunk_min = int(math.ceil(n_clip_min/self.sample_rate))

        # sampling frames from chunks
        transformed_video_all = []
        for i in range(n_chunk):
            
            n_frame_chunk = self.sample_rate*video_frame_rate
            # chunk frames
            if (i+1)*n_frame_chunk < video_length_all:
                transformed_video_i_chunk = transformed_frame_all[i*n_frame_chunk : (i+1)*n_frame_chunk]
            else:
                transformed_video_i_chunk = transformed_frame_all[i*n_frame_chunk : video_length_all]
            
            transformed_video = torch.zeros([n_frame_sample, video_channel, self.resize, self.resize])
            n_i_chunk = len(transformed_video_i_chunk)
            # sampling
            if self.sample_type == 'mid':
                mid_frame = int(n_i_chunk/2)
                if n_frame_sample < n_i_chunk:
                    transformed_video = transformed_video_i_chunk[(mid_frame - int(n_frame_sample/2)) : (mid_frame + int(n_frame_sample/2))]
                else:
                    transformed_video[ : n_i_chunk] = transformed_video_i_chunk
                    for j in range(n_i_chunk, n_frame_sample):
                        transformed_video[j] = transformed_video[n_i_chunk - 1]
            elif self.sample_type == 'uniform':
                if n_frame_sample < n_i_chunk:
                    n_interval = int(n_i_chunk/n_frame_sample)
                    transformed_video = transformed_video_i_chunk[0 : n_interval*n_frame_sample : n_interval]
                else:
                    transformed_video[ : n_i_chunk] = transformed_video_i_chunk
                    for j in range(n_i_chunk, n_frame_sample):
                        transformed_video[j] = transformed_video[n_i_chunk - 1]

            transformed_video_all.append(transformed_video)

        if n_chunk < n_chunk_min:
            for i in range(n_chunk, n_chunk_min):
                transformed_video_all.append(transformed_video_all[n_chunk - 1])
       
        return transformed_video_all, video_name_str

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str)
    parser.add_argument('--videos_dir', type=str)
    """
    dataset_file:
    LSVQ:          LSVQ_whole_train.csv, LSVQ_whole_test.csv, LSVQ_whole_test_1080p.csv
    KoNViD1k:     KoNViD-1k_data.mat
    LIVEVQC:       LIVE_VQC_data.mat
    LIVE_Qualcomm: LIVE-Qualcomm_qualcommSubjectiveData.mat
    LBVD:          LBVD_data.mat
    LIVEYTGaming:  LIVEYTGaming.mat
    youtube_ugc:   youtube_ugc_data.mat
    """
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--sample_type', type=str, default='mid')

    config = parser.parse_args()
    main(config)