
import json
from light_anime_face_detector.model import LFFDDetector


class AnimeFaceDetectionModel:
    def __init__(self, module_path, margin=0, config_path='./light_anime_face_detector/configs/anime_module.json'):
        config = json.load(open(config_path))
        self.detector = LFFDDetector(module_path, config, use_gpu=False)
        self.margin = margin
        
    def detect(self, image):
        # Img should be in cv2 format
        boxes = self.detector.detect(image)
        
        faces = []
        for box in boxes:
            face = self._fetch_box(image, box, margin=self.margin)
            faces.append(face)
            
        return faces
        
    def _fetch_box(self, img, bbox, margin=0):
        h, w, _ = img.shape
        byx = max(0, bbox['xmin'] - margin)
        tox = min(w, bbox['xmax'] + margin)
        byy = max(0, bbox['ymin'] - margin)
        toy = min(h, bbox['ymax'] + margin)
        crop = img[byy:toy,
                   byx:tox, 
                   :]
        return crop