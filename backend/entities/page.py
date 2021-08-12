from typing import Dict, List
from grouple.backend.entities.i_serializable import ISerializable


class Page(ISerializable):
    def __init__(self, pic_url: str, comments: List):
        self.pic_url = pic_url
        self.comments = comments

    def to_json(self) -> Dict[str, str]:
        page_json = dict({'pic_url': self.pic_url, 'comments': self.comments})
        return page_json