from typing import List
import json
from loguru import logger
import os
from .detector import LFFDDetector, BoundBox
from .types import Image
import os

class AnimeFaceDetectionModel:
    """Model provides face detection of anime pictures"""
    def __init__(self, margin: int = 0):
        """
        @param margin: detected face area will be enlarged with such number pixels
        """
        config = json.load(open('models/face_detection/weights/config.json'))
        self._detector = LFFDDetector(config, use_gpu=False)
        self.margin = margin

    def detect(self, image: Image) -> List[Image]:
        """Detects all faces on given cv2 BGR image"""
        bboxes = self._detector.detect(image)

        fetch_face = lambda bbox: self._fetch_box(image, bbox)
        faces = list(map(fetch_face, bboxes))
        return faces

    def _fetch_box(self, img: Image, bbox: BoundBox) -> Image:
        """Enlarges boundbox with margin pixels, crops bounded image"""
        h, w, _ = img.shape
        byx = int(max(0, bbox.xmin - self.margin))
        tox = int(min(w, bbox.xmax + self.margin))
        byy = int(max(0, bbox.ymin - self.margin))
        toy = int(min(h, bbox.ymax + self.margin))

        crop = img[byy:toy, byx:tox, :]
        return crop
