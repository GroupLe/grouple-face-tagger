from typing import Optional
import os
import hashlib
from pathlib import Path
import pickle
from collections import deque, defaultdict
from backend.entities import Manga, HashUrl

CACHE_SIZE = 100

class CacheManager:
    """
    Saves limited quantity of Manga objects. In format:
    hash_of_url/
        url.pkl        - str url of resource
        raw.pkl        - dict with parsed raw data
        ner_names.pkl  - result of NER on parsed names
        faces/         - detected faces
            i.png      - i-th detected face
    If cache reloaded, only CACHE_SIZE first items in cache folder will be loaded
    """
    def __init__(self, path: Path):
        self.cache_dir = path
        existed = os.listdir(self.cache_dir)[:CACHE_SIZE]  # todo fix leak
        self.queue = deque(existed)
        self.cache_map = defaultdict(lambda: None)
        self._warm_cache()

    def _get_hash(self, key: str) -> HashUrl:
        return hashlib.md5(key.encode()).hexdigest()

    def _warm_cache(self):
        for folder in os.listdir(self.cache_dir):

            if len(self.queue) >= CACHE_SIZE:
                return

            path = Path(self.cache_dir, folder, 'url.pkl')
            url = pickle.load(open(path, 'rb'))
            self.queue.appendleft(url)
            self.cache_map[url] = folder

    def _save_content(self, url: str, manga: Manga) -> HashUrl:
        # Returns name of folder where data stored
        hsh = self._get_hash(url)
        path = Path(self.cache_dir, hsh)
        os.mkdir(path)
        manga.dump(url, path)
        return hsh

    def get(self, url: str) -> Optional[Manga]:
        # None or content
        assert isinstance(url, str)
        folder_hashname = self.cache_map[url]

        if folder_hashname is not None:
            manga = Manga.load(Path(self.cache_dir, folder_hashname))
            return manga
        else:
            return None

    def add(self, url: str, manga: Manga) -> None:
        assert isinstance(manga, Manga)
        assert isinstance(url, str)

        if len(self.queue) >= CACHE_SIZE:
            url = self.queue.pop()
            self.cache_map.pop(url)

        self.queue.appendleft(url)
        f_name = self._save_content(url, manga)
        self.cache_map[url] = f_name
