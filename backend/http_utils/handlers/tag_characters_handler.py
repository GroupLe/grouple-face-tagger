from typing import List, Dict
from itertools import chain
from tornado.web import RequestHandler
import inject
import regex as re
import requests as r
import cv2
import numpy as np
from grouple.backend.entities import Manga
from grouple.models.NER.models.lstm.model import LSTMFixedLen as NerModel
from grouple.models.face_detection.model import AnimeFaceDetectionModel


class TagCharactersHandler(RequestHandler):

    def initialize(self, cache):
        self.cache = cache
        self.ner_model = NerModel.load('models/NER/weights/model_lstm_fixed.pt')
        self.face_detection_model = AnimeFaceDetectionModel(margin=10)

    def get(self):
        url = self.request.arguments['url'][0].decode('utf-8')

        # try fetch comments and links from cache
        manga = self.cache.get(url)
        if manga is None:
            manga = self._download_manga(url)
            manga.ner_names = self.get_ner_names(manga)
            manga.detected_faces = self.get_detected_faces(manga)
            self.cache.add(manga)

        if not manga.ner_names:
            manga.ner_names = self.get_ner_names(manga)
            self.cache.add_ner_names(manga.url, manga.ner_names)

        if not manga.detected_faces:
            manga.detected_faces = self.get_detected_faces(manga)
            self.cache.add_detected_faces(manga.url, manga.detected_faces)

        self.write({'status': 'ok',
                    'ner_names': manga.ner_names})


    def get_ner_names(self, manga: Manga) -> List[List[List[str]]]:
        ner_names = []

        for volume in manga.volumes:
            volume_names = []
            for page in volume['pages']:
                if page['comments'] != None:
                    volume_names.append(self._ner_names(page['comments'], self.ner_model))
            ner_names.append(volume_names)

        return ner_names

    def get_detected_faces(self, manga: Manga) -> List[List[List[np.array]]]:
        detected_faces = []

        for volume in manga.volumes:
            volume_faces = []
            for page in volume['pages']:
                if page['pic_url'] != None:
                    pic = r.get(page['pic_url']).content
                    decoded_pic = cv2.imdecode(np.frombuffer(pic, np.uint8), -1)
                    faces = self.face_detection_model.detect(decoded_pic)
                    volume_faces.append(faces)
            detected_faces.append(volume_faces)

        return detected_faces


    def _download_manga(self, url: str) -> Manga:
        return Manga.from_url(url)

    @inject.params(ner_model=NerModel)
    def _ner_names(self, comments: List[str], ner_model: NerModel = None) -> List[str]:
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

