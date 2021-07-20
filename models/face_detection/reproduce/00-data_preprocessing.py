from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
import os
import cv2
import regex as re
import numpy as np
from grouple.models.face_detection.model import AnimeFaceDetectionModel

def walkdir(folder):
    """Walk through every files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield str(os.path.abspath(os.path.join(dirpath, filename)))


class FacePreprocessing:

    def __init__(self, target_root):
        self.target_root = target_root

    def get_target_path_waifus(self, filepath):
        filename = Path(filepath).parts[-1]
        name = filename.replace(' ', '_')
        filename = filename.replace(' ', '_')
        name = re.sub(r"[^A-za-z]", "", name.split('.')[0])

        return name, filename

    def get_target_path_moeimouto(self, filepath):
        filename = Path(filepath).parts[-1]
        name = Path(filepath).parts[-2]
        name = re.sub(r"[^A-za-z]", "", name)

        return name[1:], filename


    def save_face(self, filepath: list):

        image = cv2.imread(filepath)

        if not (isinstance(image, np.ndarray)) or not (filepath.endswith('.png') or (filepath.endswith('.jpg'))):
            os.remove(filepath)
            return

        if len(image.shape) != 3:
            os.remove(filepath)
            return

        model = AnimeFaceDetectionModel(margin=10) #not working other way
        face = model.detect(image)

        try:
            face = np.array(face)
        except ValueError as ve:
            return

        if face.size == 0 or len(np.shape(face)) != 4 or len(face) != 1:
            return

        if 'waifus' in Path(filepath).parts:
            name, filename = self.get_target_path_waifus(filepath)
        else:
            name, filename = self.get_target_path_moeimouto(filepath)
        try:

            if not os.path.exists(Path(self.target_root) / name):
                os.mkdir(Path(self.target_root) / name)
        except FileExistsError as fee:
            pass

        target_path = self.target_root + '/' + name + '/' + filename
        os.remove(filepath)

        cv2.imwrite(target_path, face[0])


if __name__ == '__main__':

    root = Path('../../../data/face_detection/raw/')
    waifus = root / 'waifus'
    moeimouto = root / 'moeimouto-faces'
    target_root = 'C:/may/ML/GroupLe/grouple/data/face_detection/processed/'

    face_processing = FacePreprocessing(target_root)

    files1 = list(walkdir(waifus))
    files2 = list(walkdir(moeimouto))

    with Pool(processes=5) as pool:
        res1 = list(tqdm(pool.imap(face_processing.save_face, files1), total=len(files1)))
        res2 = list(tqdm(pool.imap(face_processing.save_face, files2), total=len(files2)))
