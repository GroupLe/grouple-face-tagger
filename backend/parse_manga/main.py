import sys
import os
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
from parqser.scrapper import BatchParallelScrapper
from parqser.parser import HTMLParser
from parqser.saver import CSVSaver
import examples.authenticated_session.web_components as web_components
from examples.authenticated_session.grouple_session import GroupleSession, read_spaceproxy_file

if __name__ == '__main__':
    N_JOBS = 2

    urls = ['https://readmanga.live/alice_in_murderland',
            'https://readmanga.live/buntar_liudvig',
            'https://readmanga.live/nocturnal_lover_specialty_store__bloodhound',
            'https://readmanga.live/count_cain_saga',
            'https://readmanga.live/chernaia_roza_alisy',
            'https://readmanga.live/x_day',
            'https://readmanga.live/puberty_bitter_change']


    saver = CSVSaver('parsed_info.csv')
    parser = HTMLParser.from_module(web_components)
    crawler = BatchParallelScrapper(n_jobs=N_JOBS, interval_ms=1000)
    for url_batch in crawler.batch_urls(urls):
        loaded = crawler.load_pages(url_batch)

        print(' '.join([page.status.name for page in loaded]))
        parsed = [parser.parse(page) for page in loaded]
        parsed = [page.to_dict() for page in parsed]
        saver.save_batch(parsed)
        crawler.wait()
