from typing import Optional
import os
import hashlib
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
        for manga in os.listdir(self.cache_dir):

            if len(self.queue) >= CACHE_SIZE:
                return

            path = Path(self.cache_dir, manga)
            url = Manga.load(path)['url']
            self.queue.appendleft(url)
            self.cache_map[url] = manga

    def _save_content(self, url: str) -> HashUrl:
        # Returns name of folder where data stored
        hsh = self._get_hash(url)
        path = Path(self.cache_dir, hsh)
        path = Path(str(path) + '.json')
        Manga.save(url, path)
        return hsh

    def get(self, url: str) -> Optional[Manga]:
        # None or content
        assert isinstance(url, str)
        manga_hashname = self.cache_map[url]

        if manga_hashname is not None:
            manga = Manga.load(Path(self.cache_dir, manga_hashname))

            return manga
        else:
            return None

    def add(self, url: str) -> None:
        assert isinstance(url, str)

        if len(self.queue) >= CACHE_SIZE:
            url = self.queue.pop()
            self.cache_map.pop(url)

        self.queue.appendleft(url)
        f_name = self._save_content(url)
        self.cache_map[url] = f_name


if __name__ == '__main__':
    cache = CacheManager(Path('C:/may/ML/GroupLe/grouple/data/backend/cache/cachied'))
    url = 'https://readmanga.live/buntar_liudvig'
    # cache.add(url)
    manga = cache.get(url)
    print(manga)
