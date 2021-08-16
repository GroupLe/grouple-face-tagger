import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
from tqdm import tqdm
from typing import List, Dict
from grouple.backend.parse_manga.parqser.scrapper import BatchParallelScrapper
from grouple.backend.parse_manga.parqser.parser_ import HTMLParser
import grouple.backend.parse_manga.web_components.main_page as main_web_components
import grouple.backend.parse_manga.web_components.volumes as volumes_web_components
from grouple.backend.entities.page import Page
from grouple.backend.entities.volume import Volume


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


