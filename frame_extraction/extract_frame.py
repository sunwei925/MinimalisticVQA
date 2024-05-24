import os
import pandas as pd
import cv2
import argparse
import scipy.io as scio

def extract_frame(videos_dir, video_name, size, save_folder, video_length_min):
    """
    Extract frames from a video file.
    
    Parameters:
    - videos_dir: str, path to the directory containing video files.
    - video_name: str, name of the video file (without extension).
    - save_folder: str, path to the directory where extracted frames will be saved.
    - resize: int, target size for resizing frames.
    - video_length_min: int, minimum number of frames to extract.
    """    
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]

    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Error: Couldn't open video file {filename}")
        return

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames
    
    if video_height > video_width:
        video_width_resize = size
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = size
        video_width_resize = int(video_height_resize/video_height*video_width)
        
    dim = (video_width_resize, video_height_resize)

    video_read_index = 0

    frame_idx = 0

    
    for i in range(video_length):
        has_frames, frame = cap.read()
        if has_frames:
            # key frame
            if (video_read_index < video_length) and (frame_idx % (int(video_frame_rate)) == int(video_frame_rate / 2)):
                read_frame = cv2.resize(frame, dim)
                exist_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(video_read_index) + '.png'), read_frame)        
                video_read_index += 1
            frame_idx += 1
            
    # if the length of read frame is less than video_length_min, copy the last frame
    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            exist_folder(os.path.join(save_folder, video_name_str))
            cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                    '{:03d}'.format(i) + '.png'), read_frame)


    return

def extract_frame_LBVD(videos_dir, video_name, size, save_folder, video_length_min):
    """
    Extract frames from a video file.
    
    Parameters:
    - videos_dir: str, path to the directory containing video files.
    - video_name: str, name of the video file (without extension).
    - save_folder: str, path to the directory where extracted frames will be saved.
    - resize: int, target size for resizing frames.
    - video_length_min: int, minimum number of frames to extract.
    """    
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]

    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Error: Couldn't open video file {filename}")
        return

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames
    
    if video_height > video_width:
        video_width_resize = size
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = size
        video_width_resize = int(video_height_resize/video_height*video_width)
        
    dim = (video_width_resize, video_height_resize)

    video_read_index = 0

    frame_idx = 0

    
    n_interval = int(video_length/video_length_min)
    if n_interval > 0:
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % (n_interval) == int(n_interval / 2)):
                    read_frame = cv2.resize(frame, dim)
                    exist_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                            '{:03d}'.format(video_read_index) + '.png'), read_frame)        
                    video_read_index += 1
                frame_idx += 1
                
        # if the length of read frame is less than video_length_min, copy the last frame
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                exist_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(i) + '.png'), read_frame)
    else:
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length):
                    read_frame = cv2.resize(frame, dim)
                    exist_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                            '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                    video_read_index += 1
                frame_idx += 1
                
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                exist_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(i) + '.png'), read_frame)


    return
            
def exist_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return

def main(config):

    if config.dataset == 'LSVQ_train' or config.dataset == 'LSVQ_test' or config.dataset == 'LSVQ_test_1080p': 

        dataInfo = pd.read_csv(config.dataset_file)

        video_names = dataInfo['name']
        n_video = len(video_names)
        
        for i in range(n_video):
            video_name = video_names.iloc[i] + '.mp4'
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)

    
    if config.dataset == 'KoNViD1k':

        dataInfo = scio.loadmat(config.dataset_file)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])

        for i in range(n_video):
            video_name = video_names[i]
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)
    
    elif config.dataset == 'LIVEVQC':

        dataInfo = scio.loadmat(config.dataset_file)
        n_video = len(dataInfo['video_list'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_list'][i][0][0])

        for i in range(n_video):
            video_name = video_names[i]
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)

    elif config.dataset == 'LIVE_Qualcomm':

        m = scio.loadmat(config.dataset_file)
        dataInfo = pd.DataFrame(m['qualcommVideoData'][0][0][0])
        dataInfo['MOS'] = m['qualcommSubjectiveData'][0][0][0]
        dataInfo.columns = ['file_names', 'MOS']
        dataInfo['file_names'] = dataInfo['file_names'].astype(str)
        dataInfo['file_names'] = dataInfo['file_names'].str.strip("[']")

        video_names = dataInfo['file_names'].tolist()
        
        n_video = len(video_names)

        for i in range(n_video):
            video_name = video_names[i].replace('yuv', 'mp4')
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)


    elif config.dataset == 'LBVD':

        dataInfo = scio.loadmat(config.dataset_file)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])

        for i in range(n_video):
            video_name = video_names[i]
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame_LBVD(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)

    elif config.dataset == 'LIVEYTGaming':

        dataInfo = scio.loadmat(config.dataset_file)
        n_video = len(dataInfo['video_list'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_list'][i][0][0]+'.mp4')

        for i in range(n_video):
            video_name = video_names[i]
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)
    
    elif config.dataset == 'youtube_ugc':

        dataInfo = scio.loadmat(config.dataset_file)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])

        for i in range(n_video):
            video_name = video_names[i]
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(config.videos_dir, video_name, config.resize, config.save_folder, config.video_length_min)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
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
    parser.add_argument('--videos_dir', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--video_length_min', type=int, default=8)
    parser.add_argument('--resize', type=int, default=448) # 448 for ResNet-50, 384 for Swin-B


    config = parser.parse_args()
    main(config)