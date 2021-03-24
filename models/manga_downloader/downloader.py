from webparsers.modest_parser import ModestParser
from webparsers.parsers.grouple_manga_page.parser import GroupleMangaPageParser
from webparsers.parsers.grouple_volumes_parser.parser import GroupleVolumesParser
from webparsers.sessions.empty_session import EmptySession

from pprint import pprint


class Downloader:
    def __init__(self, n_threads=1):

        pseudosave = lambda x: x
        self.manga_page_parser = ModestParser(GroupleMangaPageParser,
                                              EmptySession,
                                              pseudosave,
                                              urls=[],
                                              n_threads=n_threads)

        self.volumes_parser = ModestParser(GroupleVolumesParser,
                                           EmptySession,
                                           pseudosave,
                                           urls=[],
                                           n_threads=1)

    def download(self, url):
        base_url = '/'.join(url.split('/')[:-1])
        page_data = self.manga_page_parser.scrap_url(url)
        pprint(page_data)

        vols_comments = []
        vols_pic_urls = []

        for vol_url in page_data['volumes']:
            vol_data = self.volumes_parser.scrap_url(base_url + vol_url)
            pprint(vol_data)
            if vol_data['ok_parsed']:
                vols_comments += vol_data['comments']
                vols_pic_urls += vol_data['links']
            else:
                print('Cant parse volume')

        data = {'comments': page_data['comments'] + vols_comments,
                'description': page_data['description'],
                'pics': vols_pic_urls}

        return data
