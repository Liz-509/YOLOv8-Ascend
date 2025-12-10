import numpy as np
# from dataclasses import dataclass
from typing import List


# @dataclass
class YoloDetectResults:
    """Yolo检测结果"""
    def __init__(self):
        self.boxes: List[List[int]] = []
        self.clss: List[int] = []
        self.confs: List[float] = []
        self.masks: List[np.ndarray] = []
        self.keypoints: List[List[int, float]] = []
        self.xyxyxyxy: List[List[int]] = []
