import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
from typing import List, Dict
import json
from pathlib import Path
from grouple.backend.entities.i_serializable import ISerializable
from grouple.backend.entities.parse_manga import ParseManga
from grouple.backend.entities.volume import Volume


class Manga(ISerializable):

    def __init__(self, url, title, description, reviews, comments, volumes):
        self.url = url
        self.title = title
        self.description = description
        self.reviews = reviews
        self.comments = comments
        self.volumes = volumes
        self.ner_names = None
        self.detected_faces = None

    @property
    def ner_names(self):
        return self._ner_names

    @ner_names.setter
    def ner_names(self, names):
        self._ner_names = names

    @property
    def detected_faces(self):
        return self._detected_faces

    @detected_faces.setter
    def detected_faces(self, faces):
        self._detected_faces = faces

    @classmethod
    def from_url(cls, url: str):
        url, title, description, reviews, comments, volumes = ParseManga.parse(url)
        return cls(url, title, description, reviews, comments, volumes)

    @classmethod
    def from_json(cls, json_object):
        manga_info = json.loads(json_object)
        return cls(manga_info['url'], manga_info['title'],
                   manga_info['description'], manga_info['reviews'],
                   manga_info['comments'], manga_info['volumes'])

    def to_json(self) -> Dict:
        volume_json = dict({'url': self.url,
                            'title': self.title,
                            'description': self.description,
                            'reviews': self.reviews,
                            'comments': self.comments,
                            'volumes': self.volumes,
                            'ner_names': self.ner_names,
                            'detected_faces': self.detected_faces})
        return volume_json

    @staticmethod
    def save(url: str, path: Path) -> None:
        manga = Manga.from_url(url)
        with open(path, 'w') as file:
            json.dump(manga.to_json(), file)

    @staticmethod
    def load(path: Path) -> str:
        with open(path, 'r') as file:
            manga = json.load(file)
        return manga

    @staticmethod
    def _get_manga(url: str) -> [str, str, str, List[str], List[str],  List[Dict[str, Volume]]]:
        return ParseManga.parse(url)



if __name__ == '__main__':

    url = 'https://readmanga.live/buntar_liudvig'
    # manga = Manga.from_url(url)
    # manga_json = json.dumps(manga.to_json())
    # print(manga_json)
    path = Path('C:/may/ML/GroupLe/grouple/data/backend/processed/saved_manga/manga.json')
    Manga.save(url, path)
    manga = Manga.load(path)
    print(manga)

