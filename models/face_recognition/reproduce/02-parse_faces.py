from pathlib import Path
import pandas as pd
import os
import requests as r
from multiprocessing import Pool
from tqdm import tqdm


class ParseFaces:

    def __init__(self, target_root, main_url):
        self.target_root = target_root
        self.main_url = main_url

    def parse_faces(self, data):
        page_link = data[0]
        link = data[2]
        img_addr = data[3]
        cur_url = self.main_url + img_addr
        img = r.get(cur_url)
        cur_anime = page_link.split('/')[-1].replace('-', '_')
        anime_path = self.target_root / cur_anime
        try:
            if not os.path.exists(anime_path):
                os.mkdir(anime_path)
        except FileExistsError as fee:
            pass

        img_path = anime_path / link.split('/')[-1].replace('-', '_')
        img_file = open(str(img_path) + '.jpg', 'wb')
        img_file.write(img.content)
        img_file.close()


if __name__ == '__main__':

    path = Path('../../../data/face_detection/raw/characters/you_anime_characters_refs.csv/')
    df = pd.read_csv(path, sep='\t')
    df = df.drop_duplicates(subset='img_addr')
    main_url = df.page_link[0][:20]

    target_root = Path('../../../data/face_detection/anime_characters/')
    data = df.values.tolist()
    parse_face = ParseFaces(target_root=target_root, main_url=main_url)
    print(data[0])
    processes = 5
    with Pool(processes) as pool:
        res1 = list(tqdm(pool.imap(parse_face.parse_faces, data), total=len(data)))
