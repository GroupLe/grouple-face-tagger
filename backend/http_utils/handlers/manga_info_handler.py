from tornado.web import RequestHandler
from grouple.backend.entities import Manga


class MangaInfoHandler(RequestHandler):

    def initialize(self, cache):
        self.cache = cache

    def get(self):
        url = self.request.arguments['url'][0].decode("utf-8")

        # try fetch comments and links from cache
        manga = self.cache.get(url)
        if manga is not None:
            self.write({'status': 'ok',
                        'manga': manga.to_json()})
        else:
            manga = self._download_manga(url)
            self.cache.add(manga)

    def _download_manga(self, url: str) -> Manga:
        return Manga.from_url(url)
