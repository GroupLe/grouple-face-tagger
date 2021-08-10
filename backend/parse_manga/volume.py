from typing import Dict, List
from grouple.backend.parse_manga.i_serializable import ISerializable
from grouple.backend.parse_manga.page import Page

class Volume(ISerializable):
    def __init__(self, pages: List[Page]):
        self.pages = pages

    def to_json(self) -> Dict[str, List[Page]]:
        volume_json = dict({'pages': self.pages})
        return volume_json