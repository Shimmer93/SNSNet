from typing import List
import numpy as np
import cv2
from PIL import Image
import io
from mmpose.structures import PoseDataSample
import mmcv

def encode_mask(mask: Image):
    encoded_mask = io.BytesIO()
    mask.save(encoded_mask, format='PNG')
    encoded_mask = np.frombuffer(encoded_mask.getvalue(), dtype=np.uint8)
    return encoded_mask

def generate_jbf(pose_sample, rescale_ratio=1.0):
    if isinstance(pose_sample, PoseDataSample):
        masks = pose_sample.pred_fields.heatmaps.detach().cpu().numpy()
    else:
        masks = pose_sample

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

def encode_jbf(pose_samples: List[PoseDataSample], num_people, rescale_ratio=1.0):

    jbfs = []
    for pose_sample in pose_samples:
        masks = pose_sample.pred_fields.heatmaps.detach().cpu().numpy()
        joint_map_volume = masks[1:-1]
        body_map = masks[0:1]
        flow_map = masks[-1:]
        jbf = np.concatenate([joint_map_volume, body_map, flow_map], axis=0)
        jbfs.append(jbf)
    num_people_left = num_people - len(jbfs)
    if num_people_left > 0:
        jbf = np.zeros_like(jbfs[0])
        jbfs.extend([jbf] * num_people_left)
    jbfs = np.concatenate(jbfs, axis=0)

    h, w = jbfs.shape[-2:]
    dsize = (int(w/rescale_ratio), int(h/rescale_ratio))
    jbfs = np.stack([cv2.resize(m, dsize=dsize, interpolation=cv2.INTER_LINEAR) for m in jbfs])
    jbfs = (jbfs > 0.5).astype(np.uint8) * 255
    
    mJ, H, W = jbfs.shape
    jbfs = jbfs.reshape(mJ*H, W)
    jbfs = Image.fromarray(jbfs)
    jbfs = encode_mask(jbfs)
    
    return jbfs

def save_jbf_seq(jbf_seq, file_path):
    np.save(file_path, np.array(jbf_seq, dtype=object), allow_pickle=True)

class JBFInferenceCompact:

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
    
class JBFInferenceResize:
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