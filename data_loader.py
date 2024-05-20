import os

import pandas as pd
from PIL import Image

import torch

from torch.utils import data
import numpy as np
import scipy.io as scio
import random

class VideoDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, dataset_name, crop_size, seed=0):
        super(VideoDataset, self).__init__()

        self.video_names, self.score = self.load_video_names_scores(filename_path, dataset_name, seed)
        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.dataset_name = dataset_name

    def load_video_names_scores(self, filename_path, dataset_name, seed):
        """
        Load the names of videos and their scores in a dataset from a specified file.

        Args:
            filename_path (str): The path of the file containing the video names.
            dataset_name (str): The name of the dataset.

        Returns:
            video_names (list): A list of video names in the dataset.
            scores (list): A list of video scores in the dataset.

        Raises:
            ValueError: If the dataset name is not supported.
        """
        if 'KoNViD1k' in dataset_name or 'youtube_ugc' in dataset_name or 'LIVEVQC' in dataset_name or \
            'LBVD' in dataset_name or 'LIVEYTGaming' in dataset_name:
            dataInfo = scio.loadmat(filename_path)
            if 'KoNViD1k' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'youtube_ugc' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][0][i] for i in range(len(dataInfo['scores'][0]))]
            elif 'LIVEVQC' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0] for i in range(len(dataInfo['video_list']))]
                video_names = [dataInfo['mos'][i][0] for i in range(len(dataInfo['mos']))]
            elif 'LBVD' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                video_names = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'LIVEYTGaming' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0]+'.mp4' for i in range(len(dataInfo['video_list']))]
                video_names = [dataInfo['MOS'][i][0] for i in range(len(dataInfo['MOS']))]
            video_names, scores = self.load_subset_video_names_scores(dataset_name, video_names, scores, seed)
        elif dataset_name == 'LIVE_Qualcomm':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
            dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")
            video_names = [video_name.replace('yuv', 'mp4') for video_name in dataInfo['file_names'].tolist()]
            scores = dataInfo['MOS'].tolist()
            video_names, scores = self.load_subset_video_names_scores(dataset_name, video_names, scores, seed)
        elif dataset_name == 'LSVQ_train':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            scores = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
        elif dataset_name == 'LSVQ_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            scores = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]
        elif dataset_name == 'LSVQ_test':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()
            scores = dataInfo['mos'].tolist()
        elif dataset_name == 'LSVQ_test_1080p':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()
            scores = dataInfo['mos'].tolist()
        else:
            raise ValueError(f"Unsupported database name: {dataset_name}")
        
        return video_names, scores


    def load_subset_video_names_scores(self, dataset_name, video_names, scores, seed):
        n_videos = len(video_names)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_videos)

        if 'train' in dataset_name:
            index_subset = index_rd[ : int(n_videos * 0.6)]
        elif 'val' in dataset_name:
            index_subset = index_rd[int(n_videos * 0.6) : int(n_videos * 0.8)]
        elif 'test' in dataset_name:
            index_subset = index_rd[int(n_videos * 0.8) : ]
        else:
            raise ValueError(f"Unsupported subset database name: {dataset_name}")
        

        video_names_subset = [video_names[i] for i in index_subset]
        scores_subset = [scores[i] for i in index_subset]

        return video_names_subset, scores_subset

        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieve video data, 3D features, video scores, and video names at the specified index.

        Args:
            idx (int): The index of the video.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
                - transformed_video (torch.Tensor): A tensor of shape [video_length_read, video_channel, video_height_crop, video_width_crop]
                representing the transformed video data.
                - transformed_feature (torch.Tensor): A tensor of shape [video_length_read, 256] representing the transformed 3D features.
                - video_score (torch.Tensor): A tensor of shape [1] representing the video score.
                - video_name (str): The name of the video.

        Raises:
            None
        """
        if 'KoNViD1k' in self.dataset_name or 'LIVEVQC' in self.dataset_name or 'youtube_ugc' in self.dataset_name \
           or 'LIVEYTGaming' in self.dataset_name or 'LBVD' in self.dataset_name or 'LIVE_Qualcomm' in self.dataset_name:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif 'LSVQ' in self.dataset_name:
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]


        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       

        if 'KoNViD1k' in self.dataset_name or 'LIVEYTGaming' in self.dataset_name or 'LSVQ' in self.dataset_name \
            or 'LIVEVQC' in self.dataset_name or 'LIVEYTGaming' in self.dataset_name or 'LBVD' in self.dataset_name:
            video_length_read = 8
        elif 'LIVE_Qualcomm' in self.dataset_name:
            video_length_read = 12
        elif 'youtube_ugc' in self.dataset_name:
            video_length_read = 20

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

    
        # read 3D features
        feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name