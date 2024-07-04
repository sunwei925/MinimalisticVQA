# MinimalisticVQA
[![Platform](https://img.shields.io/badge/Platform-linux-lightgrey?logo=linux)](https://www.linux.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange?logo=python)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgree?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/MinimalisticVQA)
[![arXiv](https://img.shields.io/badge/build-paper-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2307.13981)


This is a repository for the models proposed in the paper "Analysis of video quality datasets via design of minimalistic video quality models". [TPAMI Version](https://ieeexplore.ieee.org/abstract/document/10499199) [Arxiv Version](https://arxiv.org/abs/2307.13981)

## Introduction
Blind video quality assessment (BVQA) plays an indispensable role in monitoring and improving the end-users' viewing experience in various real-world video-enabled media applications. As an experimental field, the improvements of BVQA models have been measured primarily on a few human-rated VQA datasets. Thus, it is crucial to gain a better understanding of existing VQA datasets in order to properly evaluate the current progress in BVQA. Towards this goal, we conduct a first-of-its-kind computational analysis of VQA datasets via designing minimalistic BVQA models. By minimalistic, we restrict our family of BVQA models to build only upon basic blocks: a video preprocessor (for aggressive spatiotemporal downsampling), a spatial quality analyzer, an optional temporal quality analyzer, and a quality regressor, all with the simplest possible instantiations. By comparing the quality prediction performance of different model variants on eight VQA datasets with realistic distortions, we find that nearly all datasets suffer from the easy dataset problem of varying severity, some of which even admit blind image quality assessment (BIQA) solutions. We additionally justify our claims by comparing our model generalization capabilities on these VQA datasets, and by ablating a dizzying set of BVQA design choices related to the basic building blocks. Our results cast doubt on the current progress in BVQA, and meanwhile shed light on good practices of constructing next-generation VQA datasets and models.


## Model Definitions of MinimalisticVQA
| Model | Spatial Quality Analyzer | Temporal Quality Analyzer | Weights trained on LSVQ |
| ---- |---- |---- | ---- |
|Model I | ResNet-50 (ImageNet-1k) | None | |
|Model II | ResNet-50 (pre-trained on IQA datasets) | None | |
|Model III | ResNet-50 (pre-trained on the LSVQ dataset) | None | |
|Model IV | ResNet-50 (ImageNet-1k) | SlowFast | |
|Model V | ResNet-50 (pre-trained on IQA datasets) | SlowFast | |
|Model VI | ResNet-50 (pre-trained on the LSVQ dataset) | SlowFast | |
|Model VII | Swin-B (ImageNet-1k) | None | [weights](https://www.dropbox.com/scl/fi/u2l7y5w77j85lq3108ads/MinimalisticVQA_Model_VII_LSVQ.pth?rlkey=y6vgc8cg3m6mbute68e584rus&st=c67j2jbb&dl=0) |
|Model VIII | Swin-B (pre-trained on the LSVQ dataset) | None | as above |
|Model IX | Swin-B (ImageNet-1k) | SlowFast | [weights](https://drive.google.com/file/d/1ap4uM1o2pIbVp_ODZ6kd3el1qOEQnj-k/view?usp=sharing) |
|Model X | Swin-B (pre-trained on the LSVQ dataset) | SlowFast | as above |


## Usage

### Test Datasets

- [CVD2014](https://zenodo.org/records/2646315)
- [LIVE-Qualcomm](https://live.ece.utexas.edu/research/incaptureDatabase/index.html)
- [KoNViD-1k](https://database.mmsp-kn.de/konvid-1k-database.html): The video names in the file data/KoNViD-1k_data.mat are not in the same format as those in the official released version. You can download the version of [KoNViD-1k](https://pan.baidu.com/s/1b3mMP2hHgw_8h1mnNFSNKg) (password: 1adp) that we used to match the video names.
- [LIVE-VQC](https://live.ece.utexas.edu/research/LIVEVQC/index.html)
- [YouTube-UGC](https://media.withyoutube.com/): The videos in YouTube-UGC are dynamically updated, so the videos you download may be slightly different from those used in this paper.
- [LBVD](https://github.com/cpf0079/LBVD)
- [LSVQ](https://github.com/baidut/PatchVQ): The official link may be broken; you can download the [unofficially released version](https://huggingface.co/datasets/teowu/LSVQ-videos). 
- [LIVE-YT-Gaming](https://live.ece.utexas.edu/research/LIVE-YT-Gaming/index.html) [unofficially released version](https://www.dropbox.com/scl/fi/naw09yqrommqbari8mzks/LIVE-YT-Gaming.zip?rlkey=gz556r9ft4zs0oo23f4qc3pqq&st=44k0tfxw&dl=0)

For detail introduction of these datasets, please refer to the [paper](https://arxiv.org/abs/2307.13981).


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

Download a trained model (e.g. [Model XI](https://drive.google.com/file/d/1ap4uM1o2pIbVp_ODZ6kd3el1qOEQnj-k/view?usp=sharing) trained on LSVQ). 

```
CUDA_VISIBLE_DEVICES=0 python -u test_video.py \
--model_path /home/sunwei/code/VQA/SimpleVQA/ckpts/Swin_b_384_in22k_SlowFast_Fast_LSVQ.pth \
--model_name Model_IX \
--video_name Basketball_-_RB_vs._EP_-_Jan._24_2014.mp4 \
--video_path /data/sunwei_data/LSVQ/ia-batch1 \
--resize 384 \
--video_number_min 8 \
--sample_rate 1 \
--sample_type mid \
--output logs/video_score.log \
--is_gpu
```

## Citation
**If you find this code is useful for your research, please cite**:

```latex
@article{sun2024analysis,
  title={Analysis of video quality datasets via design of minimalistic video quality models},
  author={Sun, Wei and Wen, Wen and Min, Xiongkuo and Lan, Long and Zhai, Guangtao and Ma, Kede},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```








