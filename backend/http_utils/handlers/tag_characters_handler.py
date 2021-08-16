from typing import List, Dict
from itertools import chain
from webargs.tornadoparser import use_args
from webargs import fields
from tornado.web import RequestHandler
import inject
from grouple.backend.cache import CacheManager
from grouple.backend.entities import Manga
from grouple.models.NER.models.lstm.model import LSTMFixedLen as NerModel


class TagCharactersHandler(RequestHandler):

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

        #comments = self._ner_names(manga['comments'])

    def _download_manga(self, url: str) -> Manga:
        return Manga.from_url(url)

    @inject.params(names_model=NerModel)
    def _ner_names(self, comments: List[str], names_model: NerModel = None) -> List[str]:
        # Makes NER on list of comments. Returns list of names
        names = list(map(names_model.extract_names, comments))
        names = list(chain.from_iterable(names))
        names = list(filter(lambda name: len(name) > 1, names))
        names = list(set(names))
        return names

