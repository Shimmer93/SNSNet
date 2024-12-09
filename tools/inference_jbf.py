import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch
import torch.distributed as dist
from tqdm import tqdm
import decord
from collections import OrderedDict
from glob import glob

from mmengine import load, dump
from mmengine.dist.utils import get_dist_info, init_dist
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown_batch, inference_topdown, init_model
from mmpose.utils import adapt_mmdet_pipeline, generate_jbf, save_jbf_seq, JBFInferenceCompact, JBFInferenceResize

default_det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
default_det_ckpt = '/scratch/PI/cqf/har_data/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
default_jbf_config = 'configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py'
default_jbf_ckpt = 'logs/coco_cvpr/best_coco_AP_epoch_210.pth'
default_flow_ckpt = 'logs/jhmdb_new8/best_PCK_epoch_73.pth'
default_skl_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
default_skl_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'

def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]

def det_inference(model, frames, batched=True, batch_size_det=16):
    results = []
    if batched:
        batches = [frames[i:i+batch_size_det] for i in range(0, len(frames), batch_size_det)]
        for batch in batches:
            result = inference_detector(model, batch)
            results.extend(result)
    else:
        for frame in frames:
            result = inference_detector(model, frame)
            results.append(result)

    return results

def skl_inference(anno_in, model, frames, det_results):
    anno = cp.deepcopy(anno_in)
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = anno['num_person_raw']
    kp = np.zeros((num_person, total_frames, 17, 2), dtype=np.float32)
    kp_score = np.zeros((num_person, total_frames, 17), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        pose_samples = inference_topdown(model, f, d, bbox_format='xyxy')
        for j, pose_sample in enumerate(pose_samples):
            kp[j, i] = pose_sample.pred_instances.keypoints
            kp_score[j, i] = pose_sample.pred_instances.keypoint_scores

    anno['keypoint'] = kp.astype(np.float16)
    anno['keypoint_score'] = kp_score.astype(np.float16)
    return anno

def jbf_inference(model, frames, det_results, rescale_ratio=1.0, batched=True, batch_size_pose=16):
    assert len(frames) == len(det_results)

    data = list(zip(frames, det_results))
    pose_samples = []

    if batched:
        batches = [data[i:i+batch_size_pose] for i in range(0, len(data), batch_size_pose)]
        for batch in batches:
            batch_frames, batch_det_results = zip(*batch)
            batch_pose_samples = inference_topdown_batch(model, batch_frames, batch_det_results, bbox_format='xyxy')
            pose_samples.extend(batch_pose_samples)
    else:
        for frame, det_result in data:
            pose_sample = inference_topdown(model, frame, det_result, bbox_format='xyxy')
            pose_samples.append(pose_sample)

    jbf_seq = []
    for i, pose_sample in enumerate(pose_samples):
        bbox = det_results[i]
        jbf = generate_jbf(pose_sample, bbox, rescale_ratio)
        jbf_seq.append(jbf)
    
    return jbf_seq

def load_jbf_model(jbf_config, jbf_ckpt, flow_ckpt):
    model = init_model(jbf_config, jbf_ckpt, 'cuda')
    flow_sd = torch.load(flow_ckpt, map_location='cpu')['state_dict']
    sd_backbone_flow = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('backbone_flow'):
            sd_backbone_flow[k.replace('backbone_flow.', '')] = v
    sd_flow_dec = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('head.flow_dec'):
            sd_flow_dec[k.replace('head.flow_dec.', '')] = v
    sd_flow_head = OrderedDict()
    for k, v in flow_sd.items():
        if k.startswith('head.flow_head'):
            sd_flow_head[k.replace('head.flow_head.', '')] = v
    model.backbone_flow.load_state_dict(sd_backbone_flow)
    model.head.flow_dec.load_state_dict(sd_flow_dec)
    model.head.flow_head.load_state_dict(sd_flow_head)

    return model

def load_models(det_config, det_ckpt, skl_config, skl_ckpt, jbf_config, jbf_ckpt, flow_ckpt):
    det_model = init_detector(det_config, det_ckpt, 'cuda')
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)

    skl_model = init_model(skl_config, skl_ckpt, 'cuda')
    
    jbf_model = load_jbf_model(jbf_config, jbf_ckpt, flow_ckpt)

    return det_model, skl_model, jbf_model

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--skl-config', type=str, default=default_skl_config)
    parser.add_argument('--skl-ckpt', type=str, default=default_skl_ckpt)
    parser.add_argument('--jbf-config', type=str, default=default_jbf_config)
    parser.add_argument('--jbf-ckpt', type=str, default=default_jbf_ckpt)
    parser.add_argument('--flow-ckpt', type=str, default=default_flow_ckpt)
    parser.add_argument('--video-dir', type=str, help='input video directory')
    parser.add_argument('--out-dir', type=str, help='output JBF directory')
    parser.add_argument('--rescale-ratio', type=float, help='rescale ratio for JBF', default=4.0)
    parser.add_argument('--batched', action='store_true', help='whether to use batched inference')
    parser.add_argument('--batch-size-outer', type=int, default=64)
    parser.add_argument('--batch-size-det', type=int, default=16)
    parser.add_argument('--batch-size-jbf', type=int, default=16)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
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
    
    print('Loading videos...')
    video_fns = glob(osp.join(args.video_dir, '*.mp4')) + \
                glob(osp.join(args.video_dir, '*.avi')) + \
                glob(osp.join(args.video_dir, '*.mkv')) + \
                glob(osp.join(args.video_dir, '*.mov'))
    video_fns = sorted(video_fns)
    annos = [{'filename': fn, 'frame_dir': fn.split('/')[-1].split('.')[0]} for fn in video_fns]

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

    print('Loading JBF models...')
    det_model, skl_model, jbf_model = load_models(args.det_config, args.det_ckpt, args.skl_config, args.skl_ckpt, 
                                                  args.jbf_config, args.jbf_ckpt, args.flow_ckpt)

    compact = JBFInferenceCompact(padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True)
    resize = JBFInferenceResize(scale=(256, 256), keep_ratio=False, interpolation='bilinear')

    print('Generating JBF...')
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        frame_dir = anno['frame_dir']

        batch_frames = [frames[i:i+args.batch_size_outer] for i in range(0, len(frames), args.batch_size_outer)]

        all_jbf_seqs = []
        all_new_annos = []
        all_det_results = []

        for i, batch in enumerate(batch_frames):
            det_results = det_inference(det_model, batch, batch_size_det=args.batch_size_det)
            # * Get detection results for human
            for j, det_sample in enumerate(det_results):
                # * filter boxes with small scores
                res = det_sample.pred_instances.bboxes.cpu().numpy()
                scores = det_sample.pred_instances.scores.cpu().numpy()
                res = res[scores >= args.det_score_thr]
                # * filter boxes with small areas
                box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
                assert np.all(box_areas >= 0)
                res = res[box_areas >= args.det_area_thr]
                det_results[j] = res
            all_det_results.extend(det_results)

        total_frames = len(frames)
        num_person = max([len(x) for x in all_det_results])
        anno['total_frames'] = total_frames
        anno['num_person_raw'] = num_person

        batch_det_results = [all_det_results[j:j+args.batch_size_outer] for j in range(0, len(all_det_results), args.batch_size_outer)]

        for i, (batch, det_results) in enumerate(zip(batch_frames, batch_det_results)):
            new_anno = cp.deepcopy(anno)

            shape = batch[0].shape[:2]
            new_anno['img_shape'] = shape
            new_anno = skl_inference(new_anno, skl_model, batch, det_results)

            all_new_annos.append(new_anno)
        
        anno['keypoint'] = np.concatenate([x['keypoint'] for x in all_new_annos], axis=1)
        anno['keypoint_score'] = np.concatenate([x['keypoint_score'] for x in all_new_annos], axis=1)
        anno['img_shape'] = all_new_annos[0]['img_shape']
        anno['modality'] = 'Pose'
        anno['label'] = -1
        anno['imgs'] = frames
        anno.pop('filename')

        anno = compact(anno)
        anno = resize(anno)

        frames = anno['imgs']
        batch_frames = [frames[j:j+args.batch_size_outer] for j in range(0, len(frames), args.batch_size_outer)]

        for i, (batch, det_results) in enumerate(zip(batch_frames, batch_det_results)):
            batch_next = cp.deepcopy(batch)
            batch_next.pop(0)
            batch_next.append(batch[-1] if i + 1 == len(batch_frames) else batch_frames[i+1][0])
            batch_ = np.concatenate([batch, batch_next], axis=-1)

            jbf_seq = jbf_inference(jbf_model, batch_, det_results, args.rescale_ratio, args.batched, args.batch_size_jbf)
            all_jbf_seqs.extend(jbf_seq)

        anno.pop('imgs')
        results.append(anno)
        out_fn = osp.join(args.out_dir, f'{frame_dir}.npy')
        save_jbf_seq(jbf_seq, out_fn)

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

if __name__ == '__main__':
    main()