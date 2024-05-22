# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train_BVQA.py \
# --dataset youtube_ugc \
# --model_name Model_IX \
# --datainfo data/youtube_ugc_data.mat \
# --videos_dir /data/sunwei_data/video_data/youtube_ugc/image_384p \
# --lr 0.00001 \
# --decay_ratio 0.9 \
# --decay_interval 10 \
# --print_samples 400 \
# --train_batch_size 4 \
# --num_workers 8 \
# --resize 384 \
# --crop_size 384 \
# --epochs 30 \
# --ckpt_path /data/sunwei_data/video_data/MinimalisticVQA_model/youtube_ugc/ \
# --multi_gpu \
# --n_exp 10 \
# --sample_rate 1 \
# --feature_dir /data/sunwei_data/video_data/youtube_ugc/temporal_feature_mid_sr_1 \
# >> logs/train_BVQA_youtube_ugc_Model_IX.log



# CUDA_VISIBLE_DEVICES=0,1 python -u train_BVQA.py \
# --dataset KoNViD1k \
# --model_name Model_IX \
# --datainfo data/KoNViD-1k_data.mat \
# --videos_dir /data/sunwei_data/video_data/KoNViD1k/image_384p \
# --lr 0.00001 \
# --decay_ratio 0.9 \
# --decay_interval 10 \
# --print_samples 400 \
# --train_batch_size 6 \
# --num_workers 8 \
# --resize 384 \
# --crop_size 384 \
# --epochs 30 \
# --ckpt_path /data/sunwei_data/video_data/MinimalisticVQA_model/KoNViD1k/ \
# --multi_gpu \
# --n_exp 10 \
# --sample_rate 1 \
# --feature_dir /data/sunwei_data/video_data/KoNViD1k/temporal_feature_mid_sr_1 \
# >> logs/train_BVQA_KoNViD1k_Model_IX.log

CUDA_VISIBLE_DEVICES=0,1 python -u train_BVQA.py \
--dataset LSVQ \
--model_name Model_IX \
--videos_dir /data/sunwei_data/video_data/LSVQ/image_384p \
--lr 0.00001 \
--decay_ratio 0.9 \
--decay_interval 10 \
--print_samples 1000 \
--train_batch_size 6 \
--num_workers 8 \
--resize 384 \
--crop_size 384 \
--epochs 10 \
--ckpt_path /data/sunwei_data/video_data/MinimalisticVQA_model/LSVQ/ \
--multi_gpu \
--n_exp 1 \
--sample_rate 1 \
--feature_dir /data/sunwei_data/video_data/LSVQ/temporal_feature_mid_sr_1 \
>> logs/train_BVQA_LSVQ_Model_IX.log

# CUDA_VISIBLE_DEVICES=2,3 python -u train_BVQA.py \
# --dataset LIVEVQC \
# --model_name Model_IX \
# --datainfo data/LIVE_VQC_data.mat \
# --videos_dir /data/sunwei_data/video_data/LIVEVQC/image_384p \
# --lr 0.00001 \
# --decay_ratio 0.9 \
# --decay_interval 10 \
# --print_samples 400 \
# --train_batch_size 6 \
# --num_workers 8 \
# --resize 384 \
# --crop_size 384 \
# --epochs 50 \
# --ckpt_path /data/sunwei_data/video_data/MinimalisticVQA_model/LIVEVQC/ \
# --multi_gpu \
# --n_exp 10 \
# --sample_rate 1 \
# --feature_dir /data/sunwei_data/video_data/LIVEVQC/temporal_feature_mid_sr_1 \
# >> logs/train_BVQA_LIVEVQC_Model_IX.log