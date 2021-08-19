from typing import Dict, Any, List, Union
import os
import csv
import requests as r
from pathlib import Path
from loguru import logger
from .base import BaseSaver


class ImageSaver(BaseSaver):
    def __init__(self, path: str):
        self.path = path

    def save(self, params: Dict[str, Any]):
        self.save_batch([params])

    def save_batch(self, params: List[Dict[str, Any]]):
        if len(params) == 0:
            raise AttributeError('Records batch shouldnt be empty')
        characters = params[0]

        for name in characters['characters'][0]:
            with open(Path(self.path, name + '.jpg'), 'wb') as f:
                pic = r.get(characters['characters'][0][name]).content
                f.write(pic)
