import pickle
from pathlib import Path
import os
from typing import List, AnyStr, Dict
from PIL import Image


# TODO: add description
class Manga:
    def __init__(self,
                 raw_data: Dict,
                 ner_names: List[AnyStr],
                 faces: List[Image.Image]):
        self.raw_data = raw_data
        self.ner_names = ner_names
        self.faces = faces

    def dump(self, url, path):
        pickle.dump(url, open(path.joinpath('url.pkl'), 'wb'))
        pickle.dump(self.raw_data, open(path.joinpath('raw.pkl'), 'wb'))
        pickle.dump(self.ner_names, open(path.joinpath('ner_names.pkl'), 'wb'))
        os.mkdir(path.joinpath('faces'))
        for i, file in enumerate(self.faces):
            file.save(path.joinpath(f'{i}.png'))

    def to_json(self):
        return {'recognised': {'names': self.ner_names,
                               'pics_n': len(self.faces)}}

    @classmethod
    def load(cls, path: Path):
        # url = pickle.load(open(path.joinpath('url.pkl'), 'rb'))
        raw_data = pickle.load(open(path.joinpath('raw.pkl'), 'rb'))
        ner_names = pickle.load(open(path.joinpath('ner_names.pkl'), 'rb'))
        faces = []
        for file in os.listdir(path.joinpath('faces')):
            file = Image.open(path.joinpath(f'faces/{file}.png'))
            faces.append(file)
        return cls(raw_data, ner_names, faces)