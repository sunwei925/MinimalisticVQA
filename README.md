# MinimalisticVQA
This is a repository for the models proposed in the paper "Analysis of video quality datasets via design of minimalistic video quality models". [TPAMI Version](https://ieeexplore.ieee.org/abstract/document/10499199) [Arxiv Version](https://arxiv.org/abs/2307.13981)


### Model Definitions
| Model | Spatial Quality Analyzer | Temporal Quality Analyzer |
| ---- |---- |---- |
|Model I | ResNet-50 (ImageNet-1k) | None |
|Model II | ResNet-50 (pre-trained on IQA datasets) | None |
|Model III | ResNet-50 (pre-trained on the LSVQ dataset) | None |
|Model IV | ResNet-50 (ImageNet-1k) | SlowFast |
|Model V | ResNet-50 (pre-trained on IQA datasets) | SlowFast |
|Model VI | ResNet-50 (pre-trained on the LSVQ dataset) | SlowFast |
|Model VII | Swin-B (ImageNet-1k) | None |
|Model VIII | Swin-B (pre-trained on the LSVQ dataset) | None |
|Model IX | Swin-B (ImageNet-1k) | SlowFast |
|Model X | Swin-B (pre-trained on the LSVQ dataset) | SlowFast |

### Datasets

- [CVD2014](https://zenodo.org/records/2646315)
- [LIVE-Qualcomm](https://live.ece.utexas.edu/research/incaptureDatabase/index.html)
- [KoNViD-1k](https://database.mmsp-kn.de/konvid-1k-database.html): Please note that the video names in the file data/KoNViD-1k_data.mat are not in the same format as those in the official released version. You can download the version of KoNViD-1k that we used to match the video names.
- [LIVE-VQC](https://live.ece.utexas.edu/research/LIVEVQC/index.html)
- [YouTube-UGC](https://media.withyoutube.com/): The videos in YouTube-UGC are dynamically updated, so the videos you download may be slightly different from those used in this paper.
- [LBVD](https://github.com/cpf0079/LBVD)
- [LSVQ](https://github.com/baidut/PatchVQ): The official link may be broken; you can download the [unofficially released version](https://huggingface.co/datasets/teowu/LSVQ-videos). 
- [LIVE-YT-Gaming](https://live.ece.utexas.edu/research/LIVE-YT-Gaming/index.html)

For detail introduction of these datasets, please refer to the paper. [TPAMI Version](https://ieeexplore.ieee.org/abstract/document/10499199) [Arxiv Version](https://arxiv.org/abs/2307.13981)

### Train the model
- Extract the images:
```
python -u frame_extraction/extract_frame.py \
--dataset KoNViD1k \
--dataset_file data/KoNViD-1k_data.mat \
--videos_dir /data/sunwei_data/konvid1k \
--save_folder /data/sunwei_data/video_data/KoNViD1k/image_384p \
--video_length_min 10 \
--resize 384 \
>> logs/extract_frame_KoNViD1k_384p.log
```
- Extract the temporal features:
```
CUDA_VISIBLE_DEVICES=0 python -u temporal_feature_extraction/extract_temporal_feature.py \
--dataset KoNViD1k \
--dataset_file data/KoNViD-1k_data.mat \
--videos_dir  /data/sunwei_data/konvid1k \
--feature_save_folder /data/sunwei_data/video_data/KoNViD1k/temporal_feature_mid_sr_1 \
--sample_type mid \
--sample_rate 1 \
--resize 224 \
>> logs/extract_feature_KoNViD1k_temporal_feature_mid_sr_1.log
```
- Train the model:
```
CUDA_VISIBLE_DEVICES=0,1 python -u train_BVQA.py \
--dataset KoNViD1k \
--model_name Model_IX \
--datainfo data/KoNViD-1k_data.mat \
--videos_dir /data/sunwei_data/video_data/KoNViD1k/image_384p \
--lr 0.00001 \
--decay_ratio 0.9 \
--decay_interval 10 \
--print_samples 400 \
--train_batch_size 6 \
--num_workers 8 \
--resize 384 \
--crop_size 384 \
--epochs 30 \
--ckpt_path /data/sunwei_data/video_data/MinimalisticVQA_model/KoNViD1k/ \
--multi_gpu \
--n_exp 10 \
--sample_rate 1 \
--feature_dir /data/sunwei_data/video_data/KoNViD1k/temporal_feature_mid_sr_1 \
>> logs/train_BVQA_KoNViD1k_Model_IX.log
```



### Test
Download the [Model XI](https://drive.google.com/file/d/1mcJdgYZPybUvLfWTUtZOhktsSsPlekgv/view?usp=sharing) trained on LSVQ. 

```
CUDA_VISIBLE_DEVICES=0 python -u test_video.py \
--model_path /home/sunwei/code/VQA/SimpleVQA/ckpts/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth \
--video_name Basketball_-_RB_vs._EP_-_Jan._24_2014.mp4 \
--video_path /data/sunwei_data/LSVQ/ia-batch1 \
--resize 384 \
--video_number_min 8 \
--output logs/video_score.log \
--is_gpu
```

### Citation
**If you find this code is useful for  your research, please cite**:

```latex
@article{sun2024analysis,
  title={Analysis of video quality datasets via design of minimalistic video quality models},
  author={Sun, Wei and Wen, Wen and Min, Xiongkuo and Lan, Long and Zhai, Guangtao and Ma, Kede},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

