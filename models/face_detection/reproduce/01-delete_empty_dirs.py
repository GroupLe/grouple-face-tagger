import os


def del_empty_dirs(path: str):
    for d in os.listdir(path):
        a = os.path.join(path, d)
        if os.path.isdir(a):
            del_empty_dirs(a)
            if not os.listdir(a):
                os.rmdir(a)
                print(a, 'was deleted')


if __name__ == '__main__':
    path = 'C:/may/ML/GroupLe/grouple/data/face_detection/processed'
    del_empty_dirs(path)
