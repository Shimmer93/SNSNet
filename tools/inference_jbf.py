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
from mmpose.apis import inference_topdown, inference_topdown_batch, inference_topdown_grouped, inference_topdown_batch_grouped, init_model
from mmpose.apis.inference_tracking import _track_by_iou, _track_by_oks
from mmpose.utils import adapt_mmdet_pipeline, generate_jbf, save_jbf_seq, JBFInferenceCompact, JBFInferenceResize
from mmpose.structures import bbox_xyxy2cs, bbox_cs2xyxy

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

def det_inference(model, frames, det_score_thr, det_area_thr, batched=True, batch_size_det=16):
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

    for j, det_sample in enumerate(results):
        # * filter boxes with small scores
        res = det_sample.pred_instances.bboxes.cpu().numpy()
        scores = det_sample.pred_instances.scores.cpu().numpy()
        res = res[scores >= det_score_thr]
        # * filter boxes with small areas
        box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
        assert np.all(box_areas >= 0)
        res = res[box_areas >= det_area_thr]
        results[j] = res

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

def jbf_inference_grouped(model, frames, det_results, rescale_ratio=1.0, batched=True, batch_size_pose=16):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    data = list(zip(frames, det_results))
    pose_samples = []
    bbox_list = []

    if batched:
        batches = [data[i:i+batch_size_pose] for i in range(0, len(data), batch_size_pose)]
        del frames
        for i, batch in enumerate(batches):
            batch_frames, batch_det_results = zip(*batch)
            batch_frames = list(batch_frames)
            batch_frames_next = cp.deepcopy(batch_frames)
            batch_frames_next.pop(0)
            batch_frames_next.append(batch_frames_next[-1] if i == len(batches) - 1 else batches[i+1][0][0])
            batch_frames = [np.concatenate([frame, frame_next], axis=-1) for frame, frame_next in zip(batch_frames, batch_frames_next)]
            batch_pose_samples, batch_bbox_list = inference_topdown_batch_grouped(model, batch_frames, batch_det_results, bbox_format='xyxy')
            pose_samples.extend(batch_pose_samples)
            bbox_list.extend(batch_bbox_list)
    else:
        for i, frame, det_result in enumerate(data):
            frame_next = cp.deepcopy(frame) if i == total_frames - 1 else frames[i+1]
            frame = np.concatenate([frame, frame_next], axis=-1)
            pose_sample, bbox = inference_topdown_grouped(model, frame, det_result, bbox_format='xyxy')
            pose_samples.extend(pose_sample)
            bbox_list.extend(bbox)

    jbf_seq = []
    jbf_boxes = []
    print(bbox_list[0])
    for pose_sample, bbox in zip(pose_samples, bbox_list):
        jbf = generate_jbf(pose_sample, rescale_ratio)
        jbf_seq.append(jbf)

        bbox_center, bbox_scale = bbox_xyxy2cs(bbox, padding=1.25)
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h, np.hstack([w, w]), np.hstack([h, h]))
        bbox = bbox_cs2xyxy(bbox_center, bbox_scale)
        jbf_boxes.append(bbox[0])
    
    return jbf_seq, jbf_boxes

def jbf_inference_individual(model, frames, det_results, num_person, rescale_ratio=1.0, batched=True, batch_size_pose=16):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    data = list(zip(frames, det_results))
    pose_samples_list = []

    if batched:
        batches = [data[i:i+batch_size_pose] for i in range(0, len(data), batch_size_pose)]
        for i, batch in enumerate(batches):
            batch_frames, batch_det_results = zip(*batch)
            batch_frames = list(batch_frames)
            batch_frames_next = cp.deepcopy(batch_frames)
            batch_frames_next.pop(0)
            batch_frames_next.append(batch_frames_next[-1] if i == len(batches) - 1 else batches[i+1][0][0])
            batch_frames = [np.concatenate([frame, frame_next], axis=-1) for frame, frame_next in zip(batch_frames, batch_frames_next)]
            batch_pose_samples = inference_topdown_batch(model, batch_frames, batch_det_results, bbox_format='xyxy')
            pose_samples_list.extend(batch_pose_samples)
    else:
        for i, frame, det_result in enumerate(data):
            frame_next = cp.deepcopy(frame) if i == total_frames - 1 else frames[i+1]
            frame = np.concatenate([frame, frame_next], axis=-1)
            pose_samples = inference_topdown(model, frame, det_result, bbox_format='xyxy')
            pose_samples_list.append(pose_samples)

    jbf_seq = [[] for _ in range(num_person)]
    jbf_boxes = np.zeros((num_person, total_frames, 4), dtype=np.float32)
    for i, pose_samples in enumerate(pose_samples_list):
        for j, pose_sample in enumerate(pose_samples):
            mask = pose_sample.pred_fields.heatmaps.detach().cpu().numpy()
            jbf = generate_jbf(mask, rescale_ratio)
            bbox = pose_sample.pred_instances.bboxes
            bbox_center, bbox_scale = bbox_xyxy2cs(bbox)
            w, h = np.hsplit(bbox_scale, [1])
            bbox_scale = np.where(w > h, np.hstack([w, w]), np.hstack([h, h]))
            bbox = bbox_cs2xyxy(bbox_center, bbox_scale)
            jbf_boxes[j, i] = bbox.squeeze()
            jbf_seq[j].append(jbf)
        for j in range(len(pose_samples), num_person):
            jbf_seq[j].append(generate_jbf(np.zeros_like(mask), rescale_ratio))

    return jbf_seq, jbf_boxes

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
    parser.add_argument('--mode', type=str, choices=['video_grouped', 'image_grouped', 'image_individual'], default='all_in_one',
                        help='mode for generating JBF. video_grouped: use the box of the whole video and all people in the video'
                             'image_grouped: use the box of each image and all people in the image'
                             'image_individual: use the box of each image and each person in the image')
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--skl-config', type=str, default=default_skl_config)
    parser.add_argument('--skl-ckpt', type=str, default=default_skl_ckpt)
    parser.add_argument('--jbf-config', type=str, default=default_jbf_config)
    parser.add_argument('--jbf-ckpt', type=str, default=default_jbf_ckpt)
    parser.add_argument('--flow-ckpt', type=str, default=default_flow_ckpt)
    parser.add_argument('--video-dir', type=str, help='input video directory')
    parser.add_argument('--out-dir', type=str, help='output JBF directory')
    parser.add_argument('--out-anno-dir', type=str, help='output annotation directory')
    parser.add_argument('--rescale-ratio', type=float, help='rescale ratio for JBF', default=4.0)
    parser.add_argument('--batched', action='store_true', help='whether to use batched inference')
    parser.add_argument('--batch-size-det', type=int, default=16)
    parser.add_argument('--batch-size-jbf', type=int, default=16)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
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
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        dist.barrier()
        my_part = annos[rank::world_size]

    print('Loading JBF models...')
    det_model, skl_model, jbf_model = load_models(args.det_config, args.det_ckpt, args.skl_config, args.skl_ckpt, 
                                                  args.jbf_config, args.jbf_ckpt, args.flow_ckpt)

    if args.mode == 'video_grouped':
        compact = JBFInferenceCompact(padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True)
        resize = JBFInferenceResize(scale=(256, 256), keep_ratio=False, interpolation='bilinear')
    else:
        compact = lambda x: x
        resize = lambda x: x

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_anno_dir, exist_ok=True)

    print('Generating JBF...')
    for anno in tqdm(my_part):

        # Extract frames
        frames = extract_frame(anno['filename'])
        frame_dir = anno['frame_dir']

        # Get detection results
        det_results = det_inference(det_model, frames, args.det_score_thr, args.det_area_thr, batch_size_det=args.batch_size_det)

        # Prepare annotation
        anno['total_frames'] = len(frames)
        anno['num_person_raw'] = max([len(x) for x in det_results])
        anno['img_shape'] = frames[0].shape[:2]
        anno['modality'] = 'Pose'
        anno['label'] = -1

        # Get skeleton results
        anno = skl_inference(anno, skl_model, frames, det_results)
        
        # Compact and resize for video_grouped mode
        anno['imgs'] = frames
        anno = compact(anno)
        anno = resize(anno)
        frames = anno['imgs']
        anno.pop('filename')
        anno.pop('imgs')

        # Get JBF results
        if args.mode in ['video_grouped', 'image_grouped']:
            jbf_seq, jbf_boxes = jbf_inference_grouped(jbf_model, frames, det_results, args.rescale_ratio, args.batched, args.batch_size_jbf)
            anno['jbf_boxes'] = np.stack(jbf_boxes)
            out_anno_fn = osp.join(args.out_anno_dir, f'{frame_dir}.pkl')
            dump(anno, out_anno_fn)
            out_fn = osp.join(args.out_dir, f'{frame_dir}.npy')
            save_jbf_seq(jbf_seq, out_fn)

        elif args.mode == 'image_individual':
            jbf_seq, jbf_boxes = jbf_inference_individual(jbf_model, frames, det_results, anno['num_person_raw'], args.rescale_ratio, args.batched, args.batch_size_jbf)
            for i in range(anno['num_person_raw']):
                anno_i = cp.deepcopy(anno)
                new_frame_dir_i = f'{frame_dir}_{i}'
                anno_i['frame_dir'] = new_frame_dir_i
                anno_i['jbf_boxes'] = jbf_boxes[i]
                anno_i['keypoint'] = anno['keypoint'][i:i+1]
                anno_i['keypoint_score'] = anno['keypoint_score'][i:i+1]
                jbf_seq_i = [x[i] for x in jbf_seq]
                out_anno_fn = osp.join(args.out_anno_dir, f'{new_frame_dir_i}.pkl')
                dump(anno_i, out_anno_fn)
                out_fn = osp.join(args.out_dir, f'{new_frame_dir_i}.npy')
                save_jbf_seq(jbf_seq_i, out_fn)


if __name__ == '__main__':
    main()