import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch
import torch.distributed as dist
from mmengine.dist.utils import get_dist_info, init_dist
from tqdm import tqdm
import cv2
from PIL import Image
import decord
import pickle
from collections import OrderedDict
from glob import glob
import io
import mmcv

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    # import mmpose  # noqa: F401
    from mmpose.apis import inference_topdown_batch, inference_topdown, init_model
    from mmpose.structures import PoseDataSample
    from mmpose.utils import adapt_mmdet_pipeline
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
default_det_ckpt = '/scratch/PI/cqf/har_data/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
default_jbf_config = 'configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py'
default_jbf_ckpt = 'logs/coco_cvpr/best_coco_AP_epoch_210.pth'
default_flow_ckpt = 'logs/jhmdb_new8/best_PCK_epoch_73.pth'
default_skl_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
default_skl_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'

def pkl_load(fn):
    with open (fn, 'rb') as f:
        return pickle.load(f)
    
def pkl_dump(obj, fn):
    with open (fn, 'wb') as f:
        pickle.dump(obj, f)

class MMCompact:

    def __init__(self, padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def _get_box(self, keypoint, img_shape):
        # will return x1, y1, x2, y2
        h, w = img_shape

        kp_x = keypoint[..., 0]
        kp_y = keypoint[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
        return (min_x, min_y, max_x, max_y)

    def _compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [
                np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)), mode='edge') for img in imgs
            ]
        imgs = [img[min_y: max_y, min_x: max_x] for img in imgs]
        return imgs

    def __call__(self, results):
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']
        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        min_x, min_y, max_x, max_y = self._get_box(kp, img_shape)

        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape
        results['imgs'] = self._compact_images(results['imgs'], img_shape, (min_x, min_y, max_x, max_y))
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str
    
class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "resize_size".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if 'imgs' in results:
            results['imgs'] = self._resize_imgs(results['imgs'], new_w, new_h)
        if 'keypoint' in results:
            results['keypoint'] = self._resize_kps(results['keypoint'], self.scale_factor)

        if 'gt_bboxes' in results:
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'], self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation})')
        return repr_str

def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]

def encode_mask(mask: Image):
    encoded_mask = io.BytesIO()
    mask.save(encoded_mask, format='PNG')
    encoded_mask = np.frombuffer(encoded_mask.getvalue(), dtype=np.uint8)
    return encoded_mask

def detection_inference(model, frames, batched=True, batch_size_det=16):
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

def generate_jbf(pose_sample: PoseDataSample, bbox, rescale_ratio=1.0):
    masks = pose_sample.pred_fields.heatmaps.detach().cpu().numpy()

    joint_map_volume = masks[1:-1]
    body_map = masks[0:1]
    flow_map = masks[-1:]
    jbf = np.concatenate([joint_map_volume, body_map, flow_map], axis=0)

    h, w = jbf.shape[-2:]
    dsize = (int(w/rescale_ratio), int(h/rescale_ratio))
    jbf = np.stack([cv2.resize(m, dsize=dsize, interpolation=cv2.INTER_LINEAR) for m in jbf])
    jbf = (jbf > 0.5).astype(np.uint8) * 255
    
    J, H, W = jbf.shape
    jbf = jbf.reshape(J*H, W)
    
    return dict(
        jbf=jbf,
        bbox=bbox,
        nmaps=J,
        ratio=rescale_ratio
    )

def jbf_inference(model, frames, det_results, batched=True, batch_size_pose=16):
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
        jbf = generate_jbf(pose_sample, bbox)
        jbf_seq.append(jbf)
    return jbf_seq

def load_jbf_models(det_config, det_ckpt, skl_config, skl_ckpt, jbf_config, jbf_ckpt, flow_ckpt):
    det_model = init_detector(det_config, det_ckpt, 'cuda')
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)

    skl_model = init_model(skl_config, skl_ckpt, 'cuda')
    
    jbf_model = init_model(jbf_config, jbf_ckpt, 'cuda')
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
    jbf_model.backbone_flow.load_state_dict(sd_backbone_flow)
    jbf_model.head.flow_dec.load_state_dict(sd_flow_dec)
    jbf_model.head.flow_head.load_state_dict(sd_flow_head)

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
    det_model, skl_model, jbf_model = load_jbf_models(args.pose_config, args.pose_ckpt, args.flow_ckpt)

    compact = MMCompact(padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True)
    resize = Resize(scale=(256, 256), keep_ratio=False, interpolation='bilinear')

    print('Generating JBF...')
    results = []
    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        frame_dir = anno['frame_dir']

        batch_frames = [frames[i:i+args.batch_size_outer] for i in range(0, len(frames), args.batch_size_outer)]

        all_jbf_seqs = []
        all_new_annos = []
        all_det_results = []

        for i, batch in enumerate(batch_frames):
            det_results = detection_inference(det_model, batch, batch_size_det=args.batch_size_det)
            # * Get detection results for human
            for i, det_sample in enumerate(det_results):
                # * filter boxes with small scores
                res = det_sample.pred_instances.bboxes.cpu().numpy()
                scores = det_sample.pred_instances.scores.cpu().numpy()
                res = res[scores >= args.det_score_thr]
                # * filter boxes with small areas
                box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
                assert np.all(box_areas >= 0)
                res = res[box_areas >= args.det_area_thr]
                det_results[i] = res
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
            new_anno = skl_inference(new_anno, skl_model, batch, det_results, compress=args.compress, batch_size_pose=16)

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
        results.append(anno)

        frames = anno['imgs']
        batch_frames = [frames[j:j+args.batch_size_outer] for j in range(0, len(frames), args.batch_size_outer)]

        for i, (batch, det_results) in enumerate(zip(batch_frames, all_det_results)):
            batch_next = cp.deepcopy(batch)
            batch_next.pop(0)
            batch_next.append(batch[-1] if i + 1 == len(batch_frames) else batch_frames[i+1][0])

            jbf_seq = jbf_inference(jbf_model, batch, det_results, frame_dir, args.out_dir, args.batched, args.batch_size_jbf)
            all_jbf_seqs.extend(jbf_seq)

        out_fn = osp.join(args.out_dir, f'{frame_dir}.npy')
        np.save(out_fn, np.array(jbf_seq, dtype=object), allow_pickle=True)

    print('Saving results...')
    if args.non_dist:
        pkl_dump(results, args.out)
    else:
        pkl_dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
        dist.barrier()

        if rank == 0:
            parts = [pkl_load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
            rem = len(annos) % world_size
            if rem:
                for i in range(rem, world_size):
                    parts[i].append(None)

            ordered_results = []
            for res in zip(*parts):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(annos)]
            pkl_dump(ordered_results, args.out)

if __name__ == '__main__':
    main()