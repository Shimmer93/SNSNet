# Copyright (c) OpenMMLab. All rights reserved.
from .dekr_head import DEKRHead
from .vis_head import VisPredictHead
from .yoloxpose_head import YOLOXPoseHead

__all__ = ['DEKRHead', 'VisPredictHead', 'YOLOXPoseHead']

try:
    from .rtmo_head import RTMOHead
except (ImportError, ModuleNotFoundError):
    # RTMOHead depends on MMDetection. Keep other heads importable when mmdet
    # is not installed.
    RTMOHead = None
else:
    __all__.append('RTMOHead')
