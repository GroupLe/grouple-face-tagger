from lxml import html
from .web_components import MangaPage
from .page_classifier import GroupleMangaPageClassifier
from .page_types import PageTypes


class IParser:
    def parse(self, page_str, verbose=False):
        raise NotImplemented



class GroupleMangaPageParser(IParser):
    def __init__(self):
        self.stop_symbs = ['\n', '  ']

    def parse(self, page_str, verbose=False):
        assert type(page_str) == str
        for symb in self.stop_symbs:
            page_str = page_str.replace(symb, '')

        etree_obj = html.fromstring(page_str)
        etree_obj = etree_obj.xpath("//div[@class='leftContent']")[0]
        try:
            data = MangaPage(etree_obj)
        except:
            return {'ok_parsed': False,
                    'status': PageTypes.ERR.name}

        page_type = self._get_page_status(page_str)
        return {'manga_type': data.get_manga_type(),
                'title': data.get_title(),
                'description': data.get_description(),
                'comments': data.get_comments(),
                'volumes': data.get_volumes(),
                'ok_parsed': page_type != PageTypes.CORRECT,
                'status': page_type}
    
    def _get_page_status(self, page_str):
        classifier = GroupleMangaPageClassifier(page_str)
        return classifier.get_type()
