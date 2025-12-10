import cv2

from yolov8_ascend import Yolov8Ascend
from utils.enums import ModelTask


model = Yolov8Ascend(model_path='', task=ModelTask.DET)
image = cv2.imread('')
res = model.detect(image)
