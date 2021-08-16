from typing import List, Dict
from itertools import chain
from webargs.tornadoparser import use_args
from webargs import fields
from tornado.web import RequestHandler
import inject
import regex as re
from grouple.backend.cache import CacheManager
from grouple.backend.entities import Manga
from grouple.models.NER.models.lstm.model import LSTMFixedLen as NerModel


class TagCharactersHandler(RequestHandler):

    def initialize(self, cache):
        self.cache = cache

    def get(self):
        url = self.request.arguments['url'][0].decode('utf-8')

        # try fetch comments and links from cache
        manga = self.cache.get(url)
        if manga is None:
            manga = self._download_manga(url)
            self.cache.add(manga)

        ner_model = NerModel.load('models/NER/weights/model_lstm_fixed.pt')
        ner_names = []
        for volume in manga.volumes:
            volume_names = []
            for page in volume['pages']:
                if page['comments'] != None:
                    volume_names.append(self._ner_names(page['comments'], ner_model))
            ner_names.append(volume_names)

        manga.ner_names = ner_names

        self.write({'status': 'ok',
                    'manga': manga.to_json()})

    def _download_manga(self, url: str) -> Manga:
        return Manga.from_url(url)

    @inject.params(ner_model=NerModel)
    def _ner_names(self, comments: List[str], ner_model: NerModel) -> List[str]:
        # Makes NER on list of comments. Returns list of names
        names = list(map(ner_model.extract_names, comments))
        names = list(chain.from_iterable(names))
        names = list(filter(lambda name: len(name) > 1, names))
        names = list(set(names))
        final_names = []
        for name in names:
            name = re.sub("[^А-Яа-я]", "", name)
            if name != '':
                final_names.append(name)

        return final_names

