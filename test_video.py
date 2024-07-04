import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from models import VQAModels
from pytorchvideo.models.hub import slowfast_r50
import cv2
from PIL import Image

from torchvision import transforms


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def video_processing_spatial(video_name, resize, video_number_min):

    video_name = video_name

    cap=cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print(f"Error: Couldn't open video file {video_name}")
        return

    video_channel = 3

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_length_read = max(int(video_length/video_frame_rate), video_number_min)

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames
    
    if video_height > video_width:
        video_width_resize = resize
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = resize
        video_width_resize = int(video_height_resize/video_height*video_width) 

    dim = (video_width_resize, video_height_resize)

    transformations = transforms.Compose([transforms.Resize(resize),transforms.CenterCrop(resize),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    transformed_video = torch.zeros([video_length_read, video_channel,  resize, resize])

    video_read_index = 0
    frame_idx = 0
            
    for i in range(video_length):
        has_frames, frame = cap.read()
        if has_frames:

            # key frame
            if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):
                read_frame = cv2.resize(frame, dim)

                read_frame = Image.fromarray(cv2.cvtColor(read_frame,cv2.COLOR_BGR2RGB))
                read_frame = transformations(read_frame)
                transformed_video[video_read_index] = read_frame
                video_read_index += 1

            frame_idx += 1

    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]

    cap.release()

    return transformed_video

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


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

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

def video_processing_motion(video_name, video_number_min, sample_rate, sample_type, resize):

    cap=cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print(f"Error: Couldn't open video file {video_name}")
        return

    video_channel = 3

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    # n_clip = int(video_length/video_frame_rate)
    n_clip = video_number_min
    
    n_clip_min = video_number_min

    n_frame_sample = 32
    video_length_all = n_clip * video_frame_rate

    transformed_frame_all = torch.zeros([video_length_all, video_channel, resize, resize])
    transform = transforms.Compose([transforms.Resize([resize, resize]), \
        transforms.ToTensor(), transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])

    transformed_video_all = []
    
    video_read_index = 0
    for i in range(video_length_all):
        has_frames, frame = cap.read()
        if has_frames:
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_frame = transform(read_frame)
            transformed_frame_all[video_read_index] = read_frame
            video_read_index += 1


    if video_read_index < video_length_all:
        for i in range(video_read_index, video_length_all):
            transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

    cap.release()

    # one chunk for a sample rate second video
    n_chunk = int(math.ceil(n_clip/sample_rate))
    n_chunk_min = int(math.ceil(n_clip_min/sample_rate))

    for i in range(n_chunk):
        n_frame_chunk = sample_rate*video_frame_rate
        # chunk frames
        if (i+1)*n_frame_chunk < video_length_all:
            transformed_video_i_chunk = transformed_frame_all[i*n_frame_chunk : (i+1)*n_frame_chunk]
        else:
            transformed_video_i_chunk = transformed_frame_all[i*n_frame_chunk : video_length_all]
        
        transformed_video = torch.zeros([n_frame_sample, video_channel, resize, resize])
        n_i_chunk = len(transformed_video_i_chunk)
        # sampling
        if sample_type == 'mid':
            mid_frame = int(n_i_chunk/2)
            if n_frame_sample < n_i_chunk:
                transformed_video = transformed_video_i_chunk[(mid_frame - int(n_frame_sample/2)) : (mid_frame + int(n_frame_sample/2))]
            else:
                transformed_video[ : n_i_chunk] = transformed_video_i_chunk
                for j in range(n_i_chunk, n_frame_sample):
                    transformed_video[j] = transformed_video[n_i_chunk - 1]
        elif sample_type == 'uniform':
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
    
    return transformed_video_all




def main(config):

    device = torch.device('cuda' if config.is_gpu else 'cpu')
    print('using ' + str(device))

    model_motion = slowfast()
    model_motion = model_motion.to(device)

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

    model = model.to(device=device)
    model.load_state_dict(torch.load(config.model_path))
    

    video_dist_spatial = video_processing_spatial(os.path.join(config.video_path, config.video_name), config.resize, config.video_number_min)
    video_dist_motion = video_processing_motion(os.path.join(config.video_path, config.video_name), config.video_number_min, config.sample_rate, config.sample_type, 224)
    if len(video_dist_spatial) != len(video_dist_motion):
        if len(video_dist_spatial) > len(video_dist_motion):
            video_dist_spatial = video_dist_spatial[:len(video_dist_motion)]
        else:
            video_dist_motion = video_dist_motion[:len(video_dist_spatial)]

    with torch.no_grad():
        model.eval()
        model_motion.eval()
        
        video_dist_spatial = video_dist_spatial.to(device)
        video_dist_spatial = video_dist_spatial.unsqueeze(dim=0)
        
        n_clip = len(video_dist_motion)
        feature_motion = torch.zeros([n_clip, 256])
        
        for idx, ele in enumerate(video_dist_motion):
            ele = ele.unsqueeze(dim=0)
            ele = ele.permute(0, 2, 1, 3, 4)
            ele = pack_pathway_output(ele, device)
            ele_slow_feature, ele_fast_feature = model_motion(ele)

            ele_slow_feature = ele_slow_feature.squeeze()
            ele_fast_feature = ele_fast_feature.squeeze()

            ele_feature_motion = ele_fast_feature
            ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

            feature_motion[idx] = ele_feature_motion

        feature_motion = feature_motion.to(device)
        feature_motion = feature_motion.unsqueeze(dim=0)

        outputs = model(video_dist_spatial, feature_motion)
        
        y_val = outputs.item()
        popt = [97.8954453, 10.30818116,  -0.72342544,   1.33183837]
        y_val = logistic_func(y_val, *popt)

        print('The video name: ' + config.video_name)
        print('The quality socre: {:.4f}'.format(y_val))



    output_name = config.output

    if not os.path.exists(output_name):
        os.system(r"touch {}".format(output_name))

    f = open(output_name,'w')
    f.write(config.video_name)
    f.write(',')
    f.write(str(y_val))
    f.write('\n')

    f.close()


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--video_name', type=str, default='')
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--resize', type=int)
    parser.add_argument('--video_number_min', type=int)
    parser.add_argument('--sample_rate', type=int)
    parser.add_argument('--sample_type', type=str)
    parser.add_argument('--output', type=str, default='output.txt')
    parser.add_argument('--is_gpu', action='store_true')
  

    config = parser.parse_args()

    main(config)

