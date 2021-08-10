import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
from tqdm import tqdm
from typing import List, Dict, Union
import json
from parqser.scrapper import BatchParallelScrapper
from parqser.parser_ import HTMLParser
import web_components.main_page as main_web_components
import web_components.volumes as volumes_web_components


class ISerializable:
    def __init__(self, *args, **kwargs):
        raise NotImplemented

    def to_json(self) -> Dict:
        raise NotImplemented


class Page(ISerializable):
    def __init__(self, pic_url: str, comments: List):
        self.pic_url = pic_url
        self.comments = comments

    def to_json(self) -> Dict[str, str]:
        page_json = dict({'pic_url': self.pic_url, 'comments': self.comments})
        return page_json


class Volume(ISerializable):
    def __init__(self, pages: List[Page]):
        self.pages = pages

    def to_json(self) -> Dict[str, List[Page]]:
        volume_json = dict({'pages': self.pages})
        return volume_json


class ParseManga:

    @staticmethod
    def _parse_main_page(url: str) -> Dict[str, List[str]]:

        url = [url]
        parser = HTMLParser.from_module(main_web_components)
        crawler = BatchParallelScrapper(n_jobs=2, interval_ms=1000)

        for url_batch in tqdm(crawler.batch_urls(url), total=len(url)):
            loaded = crawler.load_pages(url_batch)
            try:
                parsed = [parser.parse(page) for page in loaded]
                parsed = [page.to_dict() for page in parsed]
            except IndexError:
                pass

        return parsed[0]

    @staticmethod
    def _parse_volumes(volumes_url: List[str]) -> List[Dict[str, Volume]]:

        main_url = 'https://readmanga.live'
        urls = []
        for volume in volumes_url:
            urls.append(main_url + volume)

        parser = HTMLParser.from_module(volumes_web_components)
        scrapper = BatchParallelScrapper(n_jobs=1, interval_ms=1000)
        volumes = []
        for url_batch in tqdm(scrapper.batch_urls(urls), total=len(urls)):
            loaded = scrapper.load_pages(url_batch)

            parsed = [parser.parse(page) for page in loaded]
            parsed = [page.to_dict() for page in parsed]
            sorted_comms = dict(sorted(parsed[0]['comments_page'].items(), key=lambda x: x[0]))
            pages = []
            for i in range(len(parsed[0]['pics'])):
                pic_url = parsed[0]['pics'][i]
                try:
                    comments = sorted_comms[i]
                except KeyError:
                    comments = None
                pages.append(Page(pic_url, comments).to_json())
            volumes.append(Volume(pages).to_json())

        return volumes

    @staticmethod
    def parse(url: str) -> [str, str, str, List[str], List[str],  List[Dict[str, Volume]]]:
        main_page = ParseManga._parse_main_page(url)
        volumes = ParseManga._parse_volumes(main_page['chapters'])

        return url, main_page['title'], main_page['description'],\
               main_page['reviews'], main_page['comments'], volumes



class Manga(ISerializable):

    def __init__(self, url, title, description, reviews, comments, volumes):
        self.url = url
        self.title = title
        self.description = description
        self.reviews = reviews
        self.comments = comments
        self.volumes = volumes
        self.ner_names = None

    @classmethod
    def from_url(cls, url: str):
        url, title, description, reviews, comments, volumes = ParseManga.parse(url)
        return cls(url, title, description, reviews, comments, volumes)

    @classmethod
    def from_json(cls, json_object):
        manga_info = json.loads(json_object)
        print(manga_info)

    def to_json(self) -> Dict:
        volume_json = dict({'url': self.url,
                            'title': self.title,
                            'description': self.description,
                            'reviews': self.reviews,
                            'volumes': self.volumes,
                            'ner_names': self.ner_names})
        return volume_json

    @staticmethod
    def _get_manga(url: str) -> [str, str, str, List[str], List[str],  List[Dict[str, Volume]]]:
        return ParseManga.parse(url)



if __name__ == '__main__':

    url = 'https://readmanga.live/buntar_liudvig'
    manga = Manga.from_url(url)
    manga_json = json.dumps(manga.to_json())
    Manga.from_json(manga_json)