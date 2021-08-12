from tornado.gen import IOLoop
from tornado.web import Application
from loguru import logger
from pathlib import Path
from grouple.backend.cache import CacheManager
# from models.manga_downloader import Downloader, PicDownloader
# from models.NER import load_names_model
# from models.face_detection import AnimeFaceDetectionModel
from grouple.backend.http_utils.handlers.tag_characters_handler import TagCharactersHandler


def make_app(cache):
    args = dict(cache=cache)
    urls = [('/tag_characters', TagCharactersHandler, args)]
            #('/status', StatusHandler, args),
            #('/characters', CharactersHandler)]
    return Application(urls)

if __name__ == "__main__":
    cache = CacheManager(Path('C:/may/ML/GroupLe/grouple/data/backend/cache/cachied'))
    app = make_app(cache)
    app.listen(5022)
    logger.info('Server started')
    IOLoop.instance().start()
