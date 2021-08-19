import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
from grouple.backend.parse_manga.parqser.saver.image import ImageSaver
from grouple.backend.parse_manga.web_components.characters.characters import Characters
from grouple.backend.parse_manga.parqser.parser_.html import HTMLParser
from grouple.backend.parse_manga.parqser.scrapper.batch import BatchParallelScrapper


if __name__ == '__main__':

    url = 'https://you-anime.ru/characters?page='
    urls = []
    for i in range(2402):
        urls.append(url + str(i))

    path = '../../data/backend/characters'
    saver = ImageSaver(path)
    parser = HTMLParser.from_module(Characters())
    scrapper = BatchParallelScrapper(n_jobs=1, interval_ms=1000)

    for url_batch in scrapper.batch_urls(urls):
        loaded = scrapper.load_pages(url_batch)

        print(' '.join([page.status.name for page in loaded]))
        parsed = [parser.parse(page) for page in loaded]
        parsed = [page.to_dict() for page in parsed]
        saver.save_batch(parsed)



