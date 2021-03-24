from time import sleep
import logging
from .scrapper import load_pages


class ModestParser:
    def __init__(self, parser, session, save_batch_f, urls, proxies=None, n_threads=-1):
        """
        @param parser: class of parser
        @param session: class of session for authentification
        @param save_batch_f: function that saves data. It's params are list of data and list of urls
        @param proxies: list of proxy addresses in space_proxy site format
        @param n_threads: number of threads if loading without proxies
        """
        assert isinstance(proxies, list) or n_threads != -1
        self.save_batch_f = save_batch_f
        self.urls = urls
        self.parser = parser()
        print('Creating sessions')
        if proxies is None and n_threads == -1:
            raise Exception('For no proxy loading provide n_threads')
        elif n_threads != -1:
            proxies = [None for _ in range(n_threads)]
        self.sessions = [session(proxy) for proxy in proxies]

    def scrap_url(self, url, lock_guard=True):
        # download pages by urls
        data = load_pages([self.sessions[0]], [url])

        # parse page
        def parse(data):
            try:
                data, status = data[0]
                data = self.parser.parse(data)
                return data
            except Exception as e:
                print('Error', e, 'at url', url)
                raise e
                return {'ok_parsed': False, 'status': 'ERR'}

        parsed = parse(data)

        # if all proxy are banned
        if lock_guard:
            sleep(1.)

        return parsed


    def scrap(self, sleep_interval=1.0, ban_guard=True):
        print('Start parsing')
        for url_i in range(0, len(self.urls), len(self.sessions)):
            sleep(sleep_interval)

            urls_batch = [self.urls[i] for i in range(url_i, url_i + len(self.sessions))]
            logging.info("Start load urls %d-%d" % (url_i, url_i + len(self.sessions)))

            # download pages by urls
            data = load_pages(self.sessions, urls_batch)

            # parse page
            def parse(data, i):
                try:
                    data = self.parser.parse(data[i][0])
                    return data
                except Exception as e:
                    print('Error',e,'at url', urls_batch[i])
                    return {'ok_parsed': False, 'status': 'ERR'}

            parsed = [parse(data, i) for i in range(len(self.sessions))]
            parse_statuses = list(map(lambda d: 'OK' if d['ok_parsed'] else str(d['status']), parsed))
            print(' '.join(parse_statuses))

            # if all proxy are banned
            if ban_guard:
                if len(set(parse_statuses)) == 1 and parse_statuses[0] == 'BAN':
                    print("Stop due banned proxies")
                    break

            self.save_batch_f(parsed, urls_batch)
