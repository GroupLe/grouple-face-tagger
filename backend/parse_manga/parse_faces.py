import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
from parqser.scrapper import BatchParallelScrapper
from parqser.parser import HTMLParser
from parqser.saver import CSVSaver
from parqser.web_component import BaseComponent


class CodeLength(BaseComponent):
    def parse(self, source: str) -> int:
        source = source.encode(encoding='utf-8')
        return source

if __name__ == '__main__':
    urls = ['https://you-anime.ru/characters/osamu-dazai-2']

    saver = CSVSaver('parsed_info.csv')
    parser = HTMLParser([CodeLength()])
    scrapper = BatchParallelScrapper(n_jobs=2, interval_ms=1000)


    for url_batch in scrapper.batch_urls(urls):
        loaded = scrapper.load_pages(url_batch)

        print(' '.join([page.status.name for page in loaded]))
        parsed = [parser.parse(page) for page in loaded]
        parsed = [page.to_dict() for page in parsed]
        print(parsed[0].items())
        saver.save_batch(parsed)



