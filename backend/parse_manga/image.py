from typing import Dict, Any, List
import requests as r
from pathlib import Path
from parqser.saver.base_saver import BaseSaver


class ImageSaver(BaseSaver):
    def __init__(self, path: str):
        self.path = path

    def save(self, params: Dict[str, Any]):
        self.save_batch([params])

    def save_batch(self, params: List[Dict[str, Any]]):
        if len(params) == 0:
            raise AttributeError('Records batch shouldnt be empty')
        parameters = params[0]

        for name in parameters['characters_component'][0]:
            with open(Path(self.path, name + '.jpg'), 'wb') as f:
                pic = r.get(parameters['characters_component'][0][name]).content
                f.write(pic)
