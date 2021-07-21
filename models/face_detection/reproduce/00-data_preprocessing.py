from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
import os
import cv2
import regex as re
import numpy as np
from grouple.models.face_detection.model import AnimeFaceDetectionModel

def walkdir(folder: Path) -> str:
    """Walk through every files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield str(os.path.abspath(os.path.join(dirpath, filename)))


class FacePreprocess:

    def __init__(self, target_root: Path):
        self.target_root = target_root

    def get_target_path_waifus(self, filepath: Path) -> (str, str):
        filename = Path(filepath).parts[-1]
        name = filename.replace(' ', '_')
        filename = filename.replace(' ', '_')
        name = re.sub(r"[^A-za-z]", "", name.split('.')[0])

        return name, filename

    def get_target_path_moeimouto(self, filepath: Path) -> (str, str):
        filename = Path(filepath).parts[-1]
        name = Path(filepath).parts[-2]
        name = re.sub(r"[^A-za-z]", "", name)

        return name[1:], filename

    @staticmethod
    def is_correct_file(filepath: Path) -> bool:
        return not (filepath.endswith('.png') or (filepath.endswith('.jpg')))

    @staticmethod
    def is_correct_image(image: np.ndarray) -> bool:
        return len(np.shape(image)) != 4 or len(image) != 1

    def save_face(self, filepath: list[str]) -> _:

        image = cv2.imread(filepath)

        if not (isinstance(image, np.ndarray)) or self.is_correct_file(filepath):
            os.remove(filepath)
            return

        if len(image.shape) != 3:
            os.remove(filepath)
            return

        model = AnimeFaceDetectionModel(margin=10) #impossible to move the loading of the model outside the method
        face = model.detect(image)

        try:
            face = np.array(face)
        except ValueError:
            return

        if face.size == 0 or self.is_correct_image(face):
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




if __name__ == '__main__':

    root = Path('../../../data/face_detection/raw/')
    waifus = root / 'waifus'
    moeimouto = root / 'moeimouto-faces'
    target_root = Path('C:/may/ML/GroupLe/grouple/data/face_detection/processed/')

    face_processing = FacePreprocess(target_root)

    files1 = list(walkdir(waifus))
    files2 = list(walkdir(moeimouto))

    processes = 5
    with Pool(processes) as pool:
        res1 = list(tqdm(pool.imap(face_processing.save_face, files1), total=len(files1)))
        res2 = list(tqdm(pool.imap(face_processing.save_face, files2), total=len(files2)))
