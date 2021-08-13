from typing import Optional
import os
import hashlib
import json
from pathlib import Path
from collections import deque, defaultdict
from grouple.backend.entities import Manga, HashUrl

CACHE_SIZE = 100

class CacheManager:

    def __init__(self, path: Path):
        self.cache_dir = path
        existed = os.listdir(self.cache_dir)[:CACHE_SIZE]  # todo fix leak
        self.queue = deque(existed)
        self.cache_map = defaultdict(lambda: None)
        self._warm_cache()

    def _get_hash(self, key: str) -> HashUrl:
        return hashlib.md5(key.encode()).hexdigest()

    def _warm_cache(self):
        for hash_url in os.listdir(self.cache_dir):

            if len(self.queue) >= CACHE_SIZE:
                return

            path = Path(self.cache_dir, hash_url)
            with open(path, 'r') as file:
                s_file = file.read()
                json_manga = Manga.from_json(s_file)
            url = json_manga.url
            print(url)
            self.queue.appendleft(url)
            self.cache_map[url] = hash_url

    def _save_content(self, manga: Manga) -> HashUrl:
        # Returns name of folder where data stored
        url = manga.url
        hsh = self._get_hash(url)
        path = Path(self.cache_dir, hsh)
        path = Path(str(path) + '.json')

        with open(path, 'w') as file:
            json.dump(manga.to_json(), file)

        return hsh

    def get(self, url: str) -> Optional[Manga]:
        # None or content
        assert isinstance(url, str)
        manga_hashname = self.cache_map[url] + '.json'
        if manga_hashname is not None:
            with open(Path(self.cache_dir, manga_hashname), 'r') as file:
                s_file = file.read()
                manga = Manga.from_json(s_file)
            return manga
        else:
            return None

    def add(self, manga: Manga) -> None:
        assert isinstance(manga, Manga)
        url = manga.url

        if len(self.queue) >= CACHE_SIZE:
            url = self.queue.pop()
            self.cache_map.pop(url)

        self.queue.appendleft(url)
        f_name = self._save_content(manga)
        self.cache_map[url] = f_name


if __name__ == '__main__':
    cache = CacheManager(Path('C:/may/ML/GroupLe/grouple/data/backend/cache/cachied'))
    url = 'https://readmanga.live/buntar_liudvig'
    manga = Manga.from_url(url)
    cache.add(manga)
    manga = cache.get(url)
    print(manga.url)
