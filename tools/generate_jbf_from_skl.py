# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from tqdm import tqdm
import pickle

from mmengine import load, dump
from mmengine.dist.utils import get_dist_info, init_dist
from mmpose.utils import save_jbf_seq, JBFInferenceCompact, JBFInferenceResize

from .inference_jbf import extract_frame, jbf_inference, load_jbf_model

default_jbf_config = 'configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py'
default_jbf_ckpt = 'logs/coco_cvpr/best_coco_AP_epoch_210.pth'
default_flow_ckpt = 'logs/jhmdb_new8/best_PCK_epoch_73.pth'

def get_bboxes_from_skeletons(skls, H, W, padding=10):
    y_mins = np.min(skls[..., 1], axis=(0, -1)).astype(int)
    y_maxs = np.max(skls[..., 1], axis=(0, -1)).astype(int)
    x_mins = np.min(skls[..., 0], axis=(0, -1)).astype(int)
    x_maxs = np.max(skls[..., 0], axis=(0, -1)).astype(int)

    y_mins = np.clip(y_mins - padding, a_min=0, a_max=None)
    y_maxs = np.clip(y_maxs + padding, a_min=None, a_max=H)
    x_mins = np.clip(x_mins - padding, a_min=0, a_max=None)
    x_maxs = np.clip(x_maxs + padding, a_min=None, a_max=W)

    bboxes = np.stack([x_mins, y_mins, x_maxs, y_maxs], axis=-1)
    bboxes = np.expand_dims(bboxes, axis=1)
    bboxes = [x for x in bboxes]
    
    return bboxes

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--jbf-config', type=str, default=default_jbf_config)
    parser.add_argument('--jbf-ckpt', type=str, default=default_jbf_ckpt)
    parser.add_argument('--flow-ckpt', type=str, default=default_flow_ckpt)
    parser.add_argument('--anno-path', type=str, help='input skeleton annotation file')
    parser.add_argument('--video-dir', type=str, help='input video directory')
    parser.add_argument('--video-suffix', type=str, help='input video name suffix', default='.mp4')
    parser.add_argument('--out-dir', type=str, help='output JBF directory')
    parser.add_argument('--rescale-ratio', type=float, help='rescale ratio for JBF', default=4.0)
    parser.add_argument('--batched', action='store_true', help='whether to use batched inference')
    parser.add_argument('--batch-size-jbf', type=int, default=16)
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local-rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    print('Loading annotations...')
    with open(args.anno_path, 'rb') as f:
        annos = pickle.load(f)['annotations']
    for anno in annos:
        anno['filename'] = osp.join(args.video_dir, anno['frame_dir'] + args.video_suffix)
        anno['bboxes'] = get_bboxes_from_skeletons(anno['keypoint'], anno['img_shape'][0], anno['img_shape'][1])

    print('Initializing distributed environment...')
    if args.non_dist:
        my_part = annos
        os.makedirs(args.tmpdir, exist_ok=True)
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        if rank == 0:
            os.makedirs(args.tmpdir, exist_ok=True)
        dist.barrier()
        my_part = annos[rank::world_size]

    print('Loading JBF model...')
    model = load_jbf_model(args.jbf_config, args.jbf_ckpt, args.flow_ckpt)

    compact = JBFInferenceCompact(padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True)
    resize = JBFInferenceResize(scale=(256, 256), keep_ratio=False, interpolation='bilinear')

    print('Generating JBF...')
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for anno in tqdm(my_part):
        det_results = anno['bboxes']
        frame_dir = anno['frame_dir']
        out_fn = osp.join(args.out_dir, f'{frame_dir}.npy')
        frames = extract_frame(anno['filename'])

        anno_tmp = cp.deepcopy(anno)
        anno_tmp['imgs'] = frames
        anno_tmp = compact(anno_tmp)
        anno_tmp = resize(anno_tmp)
        frames = anno_tmp['imgs']

        frames_next = cp.deepcopy(frames)
        frames_next.pop(0)
        frames_next.append(frames[-1])
        frames = np.concatenate([frames, frames_next], axis=-1)

        # Not rigidly aligned but should be close enough
        num_frames = min(len(frames), len(det_results))
        if abs(len(frames) - len(det_results)) > 1:
            print(f'Warning: {frame_dir} frame and detection results length mismatch, {len(frames)} frames vs {len(det_results)} detection results.')
        frames = frames[:num_frames]
        det_results = det_results[:num_frames]

        jbf_seq = jbf_inference(model, frames, det_results, args.rescale_ratio, args.batched, args.batch_size_jbf)
        save_jbf_seq(jbf_seq, out_fn)

        anno_tmp['keypoint'] = anno_tmp['keypoint'].astype(np.float16)
        anno_tmp['keypoint_score'] = anno_tmp['keypoint_score'].astype(np.float16)
        anno_tmp.pop('imgs')
        anno_tmp.pop('filename')
        results.append(anno_tmp)

    print('Saving results...')
    if args.non_dist:
        dump(results, args.out)
    else:
        dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
        dist.barrier()

        if rank == 0:
            parts = [load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
            rem = len(annos) % world_size
            if rem:
                for i in range(rem, world_size):
                    parts[i].append(None)

            ordered_results = []
            for res in zip(*parts):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(annos)]
            dump(ordered_results, args.out)
            os.removedirs(args.tmpdir)

if __name__ == '__main__':
    main()