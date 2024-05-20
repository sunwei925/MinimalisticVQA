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