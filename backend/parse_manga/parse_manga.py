import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
import pandas as pd
from tqdm import tqdm
from typing import List
from parqser.scrapper import BatchParallelScrapper
from parqser.parser_ import HTMLParser
from parqser.saver import CSVSaver
import web_components.main_page as main_web_components
import web_components.volumes as volumes_web_components


class Manga:
    def __init__(self, url):
        info = self.parse_manga(url)


    def parse_main_page(self, url):

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

    def parse_volumes(self, volumes):

        main_url = 'https://readmanga.live'
        urls = []
        for volume in volumes:
            urls.append(main_url + volume)

        parser = HTMLParser.from_module(volumes_web_components)
        scrapper = BatchParallelScrapper(n_jobs=1, interval_ms=1000)

        for url_batch in tqdm(scrapper.batch_urls(urls), total=len(urls)):
            loaded = scrapper.load_pages(url_batch)

            parsed = [parser.parse(page) for page in loaded]
            parsed = [page.to_dict() for page in parsed]
            sorted_comms = sorted(parsed[0]['comments_page'].items(), key=lambda x: x[0])

        return parsed[0]['pics'], dict(sorted_comms)



    def parse_manga(self, url):
        main_page = self.parse_main_page(url)
        pics, comments = self.parse_volumes(main_page['chapters'])


        manga = {'url': url,
                 'title': main_page['title'],
                 'description': main_page['description'],
                 'reviews': main_page['reviews'],
                 'comments': main_page['comments'],
                 'volumes': [
                                {'pics': pics,
                                 'comments': comments,
                                 'names': None}
                 ]
                 }

if __name__ == '__main__':

    manga = Manga('https://readmanga.live/buntar_liudvig')

