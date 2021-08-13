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
    def url(self):
        return self._url

    @url.setter
    def url(self, url):
        self._url = url

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
    def from_json(cls, json_object: str):
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
    def _get_manga(url: str) -> [str, str, str, List[str], List[str],  List[Dict[str, Volume]]]:
        return ParseManga.parse(url)



if __name__ == '__main__':

    url = 'https://readmanga.live/buntar_liudvig'
    manga = Manga.from_url(url)
    print(manga)

