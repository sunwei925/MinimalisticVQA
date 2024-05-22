import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from models import VQAModels
from pytorchvideo.models.hub import slowfast_r50
import cv2
from PIL import Image

from torchvision import transforms

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

def video_processing_motion(video_name, video_number_min):

    cap=cv2.VideoCapture(video_name)

    video_channel = 3

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_clip = int(video_length/video_frame_rate)
    
    video_clip_min = video_number_min

    video_length_clip = 32

    transformed_frame_all = torch.zeros([video_length, video_channel, 224, 224])
    transform = transforms.Compose([transforms.Resize([224, 224]), \
        transforms.ToTensor(), transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])

    transformed_video_all = []
    
    video_read_index = 0
    for i in range(video_length):
        has_frames, frame = cap.read()
        if has_frames:
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_frame = transform(read_frame)
            transformed_frame_all[video_read_index] = read_frame
            video_read_index += 1


    if video_read_index < video_length:
        for i in range(video_read_index, video_length):
            transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

    cap.release()

    for i in range(video_clip):
        transformed_video = torch.zeros([video_length_clip, video_channel, 224, 224])
        if (i*video_frame_rate + video_length_clip) <= video_length:
            transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
        else:
            transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
            for j in range((video_length - i*video_frame_rate), video_length_clip):
                transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
        transformed_video_all.append(transformed_video)

    if video_clip < video_clip_min:
        for i in range(video_clip, video_clip_min):
            transformed_video_all.append(transformed_video_all[video_clip - 1])
    
    return transformed_video_all




def main(config):

    device = torch.device('cuda' if config.is_gpu else 'cpu')
    print('using ' + str(device))

    model_motion = slowfast()
    model_motion = model_motion.to(device)

    model = VQAModels.Model_XI()
    # model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.load_state_dict(torch.load(config.model_path))
    

    video_dist_spatial = video_processing_spatial(os.path.join(config.video_path, config.video_name), config.resize, config.video_number_min)
    video_dist_motion = video_processing_motion(os.path.join(config.video_path, config.video_name), config.video_number_min)

    with torch.no_grad():
        model.eval()
        
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
    parser.add_argument('--video_name', type=str, default='')
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--resize', type=int)
    parser.add_argument('--video_number_min', type=int)
    parser.add_argument('--output', type=str, default='output.txt')
    parser.add_argument('--is_gpu', action='store_true')
  

    config = parser.parse_args()

    main(config)