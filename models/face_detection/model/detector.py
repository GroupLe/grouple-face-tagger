from typing import List
import json
import cv2
import mxnet as mx
from .predictor import LFFDPredictor
from .types import Image


class BoundBox:
    def __init__(self, xmin: float, ymin, xmax, ymax, confidence=0):
        self.xmin = round(xmin)
        self.ymin = round(ymin)
        self.xmax = round(xmax)
        self.ymax = round(ymax)
        self.confidence = confidence

    def unpack(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


class LFFDDetector(object):
    def __init__(self, config: dict, use_gpu=False):
        lffd_config = json.load(open(config['lffd_config_path']))
        self.predictor = LFFDPredictor(
            mxnet=mx,
            symbol_file_path=config["symbol_path"],
            model_file_path=config["model_path"],
            ctx=mx.cpu() if not use_gpu else mx.gpu(0),
            **lffd_config
        )
        self.params = config['detection_parameters']

    def draw(self, image: Image, boxes: List[BoundBox]) -> Image:
        """Draw face bound boxes on the image"""
        color = (0, 255, 0)
        image = image.copy()
        for box in boxes:
            xmin, ymin, xmax, ymax = box.unpack()
            label = f"{box.confidence * 100:.2f}%"
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, thickness=1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=1)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        return image

    def detect(self, image: Image) -> List[BoundBox]:
        """Detect objects from the given BGR image (numpy array)"""
        h = image.shape[0]
        w = image.shape[1]
        resize_scale = min((self.params['size'] / max(h, w)), 1)
        bboxes, _ = self.predictor.predict(
            image,
            resize_scale=resize_scale,
            score_threshold=self.params['confidence_threshold'],
            top_k=10000,
            nms_threshold=self.params['nms_threshold'],
            nms_flag=True,
            skip_scale_branch_list=[]
        )
        return [BoundBox(*box) for box in bboxes]
