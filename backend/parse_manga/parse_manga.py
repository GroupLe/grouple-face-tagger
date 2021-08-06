import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
import pandas as pd
from tqdm import tqdm
from parqser.scrapper import BatchParallelScrapper
from parqser.parser_ import HTMLParser
from parqser.saver import CSVSaver
import web_components.volumes as web_components
from grouple_session import GroupleSession


def make_sessions(username, password, n_sessions):
    sessions = [GroupleSession() for _ in range(n_sessions)]
    for sess in sessions:
        sess.auth(username, password)
    return sessions

if __name__ == '__main__':
    N_JOBS = 5

    username, password = open('../../data/backend/secret.txt').readline().split()
    sessions = make_sessions(username, password, N_JOBS)

    df = pd.read_csv('../../data/backend/processed/parsed_info.csv', sep=',')
    main_url = 'https://readmanga.live'
    urls = []
    for chapters in df['chapters']:
        manga_chapters = eval(chapters)
        for chapter in manga_chapters:
            chapter_url = main_url + chapter
            urls.append(chapter_url)

    saver = CSVSaver('../../data/backend/processed/parsed_volumes_info.csv')
    parser = HTMLParser.from_module(web_components)
    scrapper = BatchParallelScrapper(n_jobs=2, interval_ms=1000)

    for url_batch in tqdm(scrapper.batch_urls(urls), total=len(urls)):
        loaded = scrapper.load_pages(url_batch)

        try:
            parsed = [parser.parse(page) for page in loaded]
            parsed = [page.to_dict() for page in parsed]
            saver.save_batch(parsed)
            scrapper.wait()
        except IndexError:
            pass


