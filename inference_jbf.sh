#!/bin/bash

#SBATCH --job-name=ifr_jbf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-share
#SBATCH --cpus-per-task=16
##SBATCH --nodelist=hhnode-ib-140

bash tools/dist_run.sh tools/inference_jbf.py 8 \
    --det-config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det-ckpt /scratch/PI/cqf/har_data/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --skl-config configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    --skl-ckpt https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
    --jbf-config configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py \
    --jbf-ckpt /home/zpengac/pose/PoseSegmentationMask/logs/coco_cvpr/best_coco_AP_epoch_210.pth \
    --flow-ckpt /home/zpengac/pose/PoseSegmentationMask/logs/jhmdb_new8/best_PCK_epoch_73.pth \
    --video-dir /home/zpengac/pose/PoseSegmentationMask/demo_videos \
    --out-dir /home/zpengac/pose/PoseSegmentationMask/demo_jbfs \
    --batched \
    --batch-size-outer 128 \
    --batch-size-det 16 \
    --batch-size-jbf 32 \
    --out /home/zpengac/pose/PoseSegmentationMask/demo_jbfs/demo.pkl