# Copyright (c) OpenMMLab. All rights reserved.
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
import io
import mmcv

try:
    # import mmpose  # noqa: F401
    from mmpose.apis import inference_topdown_batch, inference_topdown, init_model
    from mmpose.structures import PoseDataSample
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_jbf_config = 'configs/body_2d_keypoint/jbf/td-hm_hrnet-w32_8xb64-210e_inference-256x256.py'
default_jbf_ckpt = 'logs/coco_cvpr/best_coco_AP_epoch_210.pth'
default_flow_ckpt = 'logs/jhmdb_new8/best_PCK_epoch_73.pth'

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
    jbf = Image.fromarray(jbf)
    jbf = encode_mask(jbf)
    
    return jbf

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

    compact = MMCompact(padding=0.25, threshold=10, hw_ratio=(1., 1.), allow_imgpad=True)
    resize = Resize(scale=(256, 256), keep_ratio=False, interpolation='bilinear')

    print('Generating JBF...')
    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for anno in tqdm(my_part):
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

        det_results = anno['bboxes']
        frame_dir = anno['frame_dir']
        jbf_seq = jbf_inference(model, frames, det_results, args.rescale_ratio, args.batched, args.batch_size_jbf)
        
        out_fn = osp.join(args.out_dir, f'{frame_dir}.npy')
        np.save(out_fn, np.array(jbf_seq, dtype=object), allow_pickle=True)

        anno_tmp.pop('imgs')
        anno_tmp.pop('filename')
        results.append(anno_tmp)

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