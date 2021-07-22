import os
import cv2


def brg2rgb(path: str) -> None:
    for dirpath, dirs, files in os.walk(path):
        for filename in files:
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            os.remove(image_path)
            cv2.imwrite(image_path, image)


if __name__ == '__main__':
    path = 'C:/may/ML/GroupLe/grouple/data/face_detection/processed'
    brg2rgb(path)
