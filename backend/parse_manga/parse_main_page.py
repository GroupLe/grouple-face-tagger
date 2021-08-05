import sys
sys.path.insert(0, 'C:/may/ML/GroupLe/grouple/backend/parse_manga/parqser')
import pandas as pd
from tqdm import tqdm
from parqser.scrapper import BatchParallelScrapper
from parqser.parser import HTMLParser
from parqser.saver import CSVSaver
import web_components.main_page as web_components


if __name__ == '__main__':
    N_JOBS = 5

    df = pd.read_csv('../../data/backend/raw/links.csv', sep=',')
    urls = df.link.tolist()

    saver = CSVSaver('../../data/backend/processed/parsed_info.csv')
    parser = HTMLParser.from_module(web_components)
    crawler = BatchParallelScrapper(n_jobs=N_JOBS, interval_ms=1000)
    for url_batch in tqdm(crawler.batch_urls(urls), total=len(urls)):
        loaded = crawler.load_pages(url_batch)
        try:
            parsed = [parser.parse(page) for page in loaded]
            parsed = [page.to_dict() for page in parsed]
            saver.save_batch(parsed)
            crawler.wait()
        except IndexError:
            pass


