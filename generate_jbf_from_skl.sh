#!/bin/bash

#SBATCH --job-name=gen_jbf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-share
#SBATCH --cpus-per-task=16
##SBATCH --nodelist=hhnode-ib-140

# bash tools/dist_run.sh tools/generate_jbf_from_skl.py 8 \
#     --jbf-config configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py \
#     --jbf-ckpt /home/zpengac/pose/PoseSegmentationMask/logs/coco_cvpr/best_coco_AP_epoch_210.pth \
#     --flow-ckpt /home/zpengac/pose/SNSNet/checkpoints/flow_ckpt.pth \
#     --anno-path /scratch/PI/cqf/har_data/pkls/hmdb51_hrnet.pkl \
#     --video-dir /scratch/PI/cqf/har_data/hmdb51/videos \
#     --video-suffix .avi \
#     --out-dir /scratch/PI/cqf/har_data/hmdb51/jbf \
#     --batched \
#     --batch-size-jbf 32 \
#     --out /scratch/PI/cqf/har_data/pkls/hmdb51_hrnet_tmp.pkl

# python tools/merge_jbf.py --orig_pkl /scratch/PI/cqf/har_data/pkls/hmdb51_hrnet.pkl --jbf_pkl /scratch/PI/cqf/har_data/pkls/hmdb51_hrnet_tmp.pkl --out_pkl /scratch/PI/cqf/har_data/pkls/hmdb51_hrnet_jbf.pkl
# rm /scratch/PI/cqf/har_data/pkls/ntu120_hrnet_tmp.pkl

bash tools/dist_run.sh tools/generate_jbf_from_skl.py 1 \
    --jbf-config configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py \
    --jbf-ckpt /home/zpengac/pose/PoseSegmentationMask/logs/coco_cvpr/best_coco_AP_epoch_210.pth \
    --flow-ckpt /home/zpengac/pose/SNSNet/checkpoints/flow_ckpt.pth \
    --anno-path /scratch/PI/cqf/har_data/pkls/ntu120_hrnet.pkl \
    --video-dir /scratch/PI/cqf/har_data/ntu/nturgb+d_rgb \
    --video-suffix _rgb.avi \
    --out-dir /scratch/PI/cqf/har_data/ntu/jbf \
    --batched \
    --batch-size-jbf 32 \
    --out /scratch/PI/cqf/har_data/pkls/ntu120_hrnet_tmp.pkl

# python tools/merge_pkl.py --orig_pkl /scratch/PI/cqf/har_data/pkls/ntu120_hrnet.pkl --jbf_pkl /scratch/PI/cqf/har_data/pkls/ntu120_hrnet_tmp.pkl --out_pkl /scratch/PI/cqf/har_data/pkls/ntu120_hrnet_jbf.pkl
# rm /scratch/PI/cqf/har_data/pkls/ntu120_hrnet_tmp.pkl