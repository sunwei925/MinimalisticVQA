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
            'LBVD' in dataset_name or 'LIVEYTGaming' in dataset_name or 'CVD2014' in dataset_name:
            dataInfo = scio.loadmat(filename_path)
            if 'KoNViD1k' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'youtube_ugc' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][0][i] for i in range(len(dataInfo['scores'][0]))]
            elif 'LIVEVQC' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0] for i in range(len(dataInfo['video_list']))]
                scores = [dataInfo['mos'][i][0] for i in range(len(dataInfo['mos']))]
            elif 'LBVD' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'LIVEYTGaming' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0]+'.mp4' for i in range(len(dataInfo['video_list']))]
                scores = [dataInfo['MOS'][i][0] for i in range(len(dataInfo['MOS']))]
            elif 'CVD2014' in dataset_name:
                video_folder = [dataInfo['video_folder'][i][0][0] for i in range(len(dataInfo['video_folder']))]
                video_names = [video_folder[i][:4]+video_folder[i][5]+'/'+dataInfo['video_name'][i][0][0]+'.avi' for i in range(len(dataInfo['video_name']))]
                scores = [dataInfo['video_score'][i][0] for i in range(len(dataInfo['video_score']))]
            if 'train' in dataset_name or 'val' in dataset_name or 'test' in dataset_name:
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
            if 'train' in dataset_name or 'val' in dataset_name or 'test' in dataset_name:
                video_names, scores = self.load_subset_video_names_scores(dataset_name, video_names, scores, seed)           
        elif dataset_name == 'LSVQ_train_all' or dataset_name == 'LSVQ_test' or dataset_name == 'LSVQ_test_1080p':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()
            scores = dataInfo['mos'].tolist()
        elif dataset_name == 'LSVQ_train':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            scores = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
        elif dataset_name == 'LSVQ_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            scores = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]
        elif dataset_name == 'KonVid150k_train_all' or dataset_name == 'KonVid150k_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['video_name'].tolist()
            scores = dataInfo['video_score'].tolist()
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
           or 'LIVEYTGaming' in self.dataset_name or 'LBVD' in self.dataset_name or 'LIVE_Qualcomm' in self.dataset_name\
            or 'CVD2014' in self.dataset_name or 'KonVid150k' in self.dataset_name:
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
        elif 'KonVid150k' in self.dataset_name:
            video_length_read = 5
        elif 'CVD2014' in self.dataset_name:
            video_length_read = 10
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










class VideoDataset_pair(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, dataset_name, crop_size, seed=0):
        super(VideoDataset_pair, self).__init__()

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
            'LBVD' in dataset_name or 'LIVEYTGaming' in dataset_name or 'CVD2014' in dataset_name:
            dataInfo = scio.loadmat(filename_path)
            if 'KoNViD1k' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'youtube_ugc' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][0][i] for i in range(len(dataInfo['scores'][0]))]
            elif 'LIVEVQC' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0] for i in range(len(dataInfo['video_list']))]
                scores = [dataInfo['mos'][i][0] for i in range(len(dataInfo['mos']))]
            elif 'LBVD' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'LIVEYTGaming' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0]+'.mp4' for i in range(len(dataInfo['video_list']))]
                scores = [dataInfo['MOS'][i][0] for i in range(len(dataInfo['MOS']))]
            elif 'CVD2014' in dataset_name:
                video_folder = [dataInfo['video_folder'][i][0][0] for i in range(len(dataInfo['video_folder']))]
                video_names = [video_folder[i][:4]+video_folder[i][5]+'/'+dataInfo['video_name'][i][0][0]+'.avi' for i in range(len(dataInfo['video_name']))]
                scores = [dataInfo['video_score'][i][0] for i in range(len(dataInfo['video_score']))]
            if 'train' in dataset_name or 'val' in dataset_name or 'test' in dataset_name:
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
            if 'train' in dataset_name or 'val' in dataset_name or 'test' in dataset_name:
                video_names, scores = self.load_subset_video_names_scores(dataset_name, video_names, scores, seed)           
        elif dataset_name == 'LSVQ_train_all' or dataset_name == 'LSVQ_test' or dataset_name == 'LSVQ_test_1080p':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()
            scores = dataInfo['mos'].tolist()
        elif dataset_name == 'LSVQ_train':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            scores = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
        elif dataset_name == 'LSVQ_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            scores = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]
        elif dataset_name == 'KonVid150k_train_all' or dataset_name == 'KonVid150k_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['video_name'].tolist()
            scores = dataInfo['video_score'].tolist()
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

        idx_second = random.randint(0, self.length - 1)
        if idx == idx_second:
            idx_second = (idx_second + 1) % self.length
        while self.score[idx] == self.score[idx_second]:
            idx_second = random.randint(0, self.length - 1)
            if idx == idx_second:
                idx_second = (idx_second + 1) % self.length

        if 'KoNViD1k' in self.dataset_name or 'LIVEVQC' in self.dataset_name or 'youtube_ugc' in self.dataset_name \
           or 'LIVEYTGaming' in self.dataset_name or 'LBVD' in self.dataset_name or 'LIVE_Qualcomm' in self.dataset_name\
            or 'CVD2014' in self.dataset_name or 'KonVid150k' in self.dataset_name:
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]

            video_name_second = self.video_names[idx_second]
            video_name_str_second = video_name_second[:-4]
        elif 'LSVQ' in self.dataset_name:
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]

            video_name_second = self.video_names[idx_second] + '.mp4'
            video_name_str_second = video_name_second[:-4]


        video_score = torch.FloatTensor(np.array(float(self.score[idx])))
        video_score_second = torch.FloatTensor(np.array(float(self.score[idx_second])))

        path_name = os.path.join(self.videos_dir, video_name_str)
        path_name_second = os.path.join(self.videos_dir, video_name_str_second)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       

        if 'KoNViD1k' in self.dataset_name or 'LIVEYTGaming' in self.dataset_name or 'LSVQ' in self.dataset_name \
            or 'LIVEVQC' in self.dataset_name or 'LIVEYTGaming' in self.dataset_name or 'LBVD' in self.dataset_name:
            video_length_read = 8
        elif 'KonVid150k' in self.dataset_name:
            video_length_read = 5
        elif 'CVD2014' in self.dataset_name:
            video_length_read = 10
        elif 'LIVE_Qualcomm' in self.dataset_name:
            video_length_read = 12
        elif 'youtube_ugc' in self.dataset_name:
            video_length_read = 20

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        transformed_video_second = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame


        for i in range(video_length_read):
            imge_name_second = os.path.join(path_name_second, '{:03d}'.format(i) + '.png')
            read_frame_second = Image.open(imge_name_second)
            read_frame_second = read_frame_second.convert('RGB')
            read_frame_second = self.transform(read_frame_second)
            transformed_video_second[i] = read_frame_second

    
        # read 3D features
        feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
        transformed_feature = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature[i] = feature_3D

        feature_folder_name_second = os.path.join(self.data_dir_3D, video_name_str_second)
        transformed_feature_second = torch.zeros([video_length_read, 256])
        for i in range(video_length_read):
            feature_3D_second = np.load(os.path.join(feature_folder_name_second, 'feature_' + str(int(i)) + '_fast_feature.npy'))
            feature_3D_second = torch.from_numpy(feature_3D_second)
            feature_3D_second = feature_3D_second.squeeze()
            transformed_feature_second[i] = feature_3D_second

        return transformed_video, transformed_feature, video_score, video_name, transformed_video_second, transformed_feature_second, video_score_second, video_name_second




class VideoDataset_pair_average_sampling(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, dataset_information, transform, crop_size, seed=0):
        super(VideoDataset_pair_average_sampling, self).__init__()

        self.video_names = []
        self.scores = []
        self.videos_dir = []
        self.data_dir_3D = []
        self.n_dataset = len(dataset_information)
        self.dataset_names = [dataset_information[i_dataset]['dataset_name'] for i_dataset in dataset_information]
        for i_dataset in dataset_information:
            video_names, score = self.load_video_names_scores(dataset_information[i_dataset]['datainfo'], dataset_information[i_dataset]['dataset_name']+'_train', seed)
            self.video_names.append(video_names)
            self.scores.append(score)
            self.videos_dir.append(dataset_information[i_dataset]['videos_dir'])
            self.data_dir_3D.append(dataset_information[i_dataset]['feature_dir'])

        self.crop_size = crop_size
        self.transform = transform
        self.length_all = [len(video_names) for video_names in self.video_names]
        self.length = max(self.length_all)

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
            'LBVD' in dataset_name or 'LIVEYTGaming' in dataset_name or 'CVD2014' in dataset_name:
            dataInfo = scio.loadmat(filename_path)
            if 'KoNViD1k' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'youtube_ugc' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][0][i] for i in range(len(dataInfo['scores'][0]))]
            elif 'LIVEVQC' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0] for i in range(len(dataInfo['video_list']))]
                scores = [dataInfo['mos'][i][0] for i in range(len(dataInfo['mos']))]
            elif 'LBVD' in dataset_name:
                video_names = [dataInfo['video_names'][i][0][0] for i in range(len(dataInfo['video_names']))]
                scores = [dataInfo['scores'][i][0] for i in range(len(dataInfo['scores']))]
            elif 'LIVEYTGaming' in dataset_name:
                video_names = [dataInfo['video_list'][i][0][0]+'.mp4' for i in range(len(dataInfo['video_list']))]
                scores = [dataInfo['MOS'][i][0] for i in range(len(dataInfo['MOS']))]
            elif 'CVD2014' in dataset_name:
                video_folder = [dataInfo['video_folder'][i][0][0] for i in range(len(dataInfo['video_folder']))]
                video_names = [video_folder[i][:4]+video_folder[i][5]+'/'+dataInfo['video_name'][i][0][0]+'.avi' for i in range(len(dataInfo['video_name']))]
                scores = [dataInfo['video_score'][i][0] for i in range(len(dataInfo['video_score']))]
            if 'train' in dataset_name or 'val' in dataset_name or 'test' in dataset_name:
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
            if 'train' in dataset_name or 'val' in dataset_name or 'test' in dataset_name:
                video_names, scores = self.load_subset_video_names_scores(dataset_name, video_names, scores, seed)           
        elif dataset_name == 'LSVQ_train_all' or dataset_name == 'LSVQ_test' or dataset_name == 'LSVQ_test_1080p':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()
            scores = dataInfo['mos'].tolist()
        elif dataset_name == 'LSVQ_train':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[:int(len(dataInfo) * 0.8)]
            scores = dataInfo['mos'].tolist()[:int(len(dataInfo) * 0.8)]
        elif dataset_name == 'LSVQ_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['name'].tolist()[int(len(dataInfo) * 0.8):]
            scores = dataInfo['mos'].tolist()[int(len(dataInfo) * 0.8):]
        elif dataset_name == 'KonVid150k_train_all' or dataset_name == 'KonVid150k_val':
            dataInfo = pd.read_csv(filename_path)
            video_names = dataInfo['video_name'].tolist()
            scores = dataInfo['video_score'].tolist()
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

        transformed_video_all = []
        transformed_feature_all = []
        video_score_all = []
        video_name_all = []
        transformed_video_second_all = []
        transformed_feature_second_all = []
        video_score_second_all = []
        video_name_second_all = []
        for i_dataset in range(self.n_dataset):
            if idx >= self.length_all[i_dataset]:
                idx = idx % self.length_all[i_dataset]

            idx_second = random.randint(0, self.length_all[i_dataset] - 1)
            if idx == idx_second:
                idx_second = (idx_second + 1) % self.length_all[i_dataset]
            while self.scores[i_dataset][idx] == self.scores[i_dataset][idx_second]:
                idx_second = random.randint(0, self.length_all[i_dataset] - 1)
                if idx == idx_second:
                    idx_second = (idx_second + 1) % self.length_all[i_dataset]

            dataset_name = self.dataset_names[i_dataset]
            if 'KoNViD1k' in dataset_name or 'LIVEVQC' in dataset_name or 'youtube_ugc' in dataset_name \
            or 'LIVEYTGaming' in dataset_name or 'LBVD' in dataset_name or 'LIVE_Qualcomm' in dataset_name\
                or 'CVD2014' in dataset_name or 'KonVid150k' in dataset_name:
                video_name = self.video_names[i_dataset][idx]
                video_name_str = video_name[:-4]

                video_name_second = self.video_names[i_dataset][idx_second]
                video_name_str_second = video_name_second[:-4]
            elif 'LSVQ' in dataset_name:
                video_name = self.video_names[i_dataset][idx] + '.mp4'
                video_name_str = video_name[:-4]

                video_name_second = self.video_names[i_dataset][idx_second] + '.mp4'
                video_name_str_second = video_name_second[:-4]


            video_score = torch.FloatTensor(np.array(float(self.scores[i_dataset][idx])))
            video_score = video_score.unsqueeze(0)
            video_score_all.append(video_score)
            video_score_second = torch.FloatTensor(np.array(float(self.scores[i_dataset][idx_second])))
            video_score_second = video_score_second.unsqueeze(0)
            video_score_second_all.append(video_score_second)

            path_name = os.path.join(self.videos_dir[i_dataset], video_name_str)
            path_name_second = os.path.join(self.videos_dir[i_dataset], video_name_str_second)

            video_channel = 3

            video_height_crop = self.crop_size
            video_width_crop = self.crop_size
        

            if 'KoNViD1k' in dataset_name or 'LIVEYTGaming' in dataset_name or 'LSVQ' in dataset_name \
                or 'LIVEVQC' in dataset_name or 'LIVEYTGaming' in dataset_name or 'LBVD' in dataset_name:
                video_length_read = 4
            elif 'KonVid150k' in dataset_name:
                video_length_read = 5
            elif 'CVD2014' in dataset_name:
                video_length_read = 10
            elif 'LIVE_Qualcomm' in dataset_name:
                video_length_read = 12
            elif 'youtube_ugc' in dataset_name:
                video_length_read = 20

            transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
            transformed_video_second = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


            for i in range(video_length_read):
                imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_video[i] = read_frame
            transformed_video = transformed_video.unsqueeze(0)
            transformed_video_all.append(transformed_video)


            for i in range(video_length_read):
                imge_name_second = os.path.join(path_name_second, '{:03d}'.format(i) + '.png')
                read_frame_second = Image.open(imge_name_second)
                read_frame_second = read_frame_second.convert('RGB')
                read_frame_second = self.transform(read_frame_second)
                transformed_video_second[i] = read_frame_second
            transformed_video_second = transformed_video_second.unsqueeze(0)
            transformed_video_second_all.append(transformed_video_second)

        
            # read 3D features
            feature_folder_name = os.path.join(self.data_dir_3D[i_dataset], video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(int(i)) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
            transformed_feature = transformed_feature.unsqueeze(0)
            transformed_feature_all.append(transformed_feature)

            feature_folder_name_second = os.path.join(self.data_dir_3D[i_dataset], video_name_str_second)
            transformed_feature_second = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                feature_3D_second = np.load(os.path.join(feature_folder_name_second, 'feature_' + str(int(i)) + '_fast_feature.npy'))
                feature_3D_second = torch.from_numpy(feature_3D_second)
                feature_3D_second = feature_3D_second.squeeze()
                transformed_feature_second[i] = feature_3D_second
            transformed_feature_second = transformed_feature_second.unsqueeze(0)
            transformed_feature_second_all.append(transformed_feature_second)












        transformed_video_all = torch.cat(transformed_video_all, 0)
        transformed_feature_all = torch.cat(transformed_feature_all, 0)
        video_score_all = torch.cat(video_score_all, 0)
        transformed_video_second_all = torch.cat(transformed_video_second_all, 0)
        transformed_feature_second_all = torch.cat(transformed_feature_second_all, 0)
        video_score_second_all = torch.cat(video_score_second_all, 0)

        return transformed_video_all, transformed_feature_all, video_score_all, video_name_all, transformed_video_second_all, transformed_feature_second_all, video_score_second_all, video_name_second_all