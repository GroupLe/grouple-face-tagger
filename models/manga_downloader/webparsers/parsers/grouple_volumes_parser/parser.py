from lxml import html
from .web_components import Volume
from .page_classifier import GroupleVolumesClassifier
from .page_types import PageTypes


class IParser:
    def parse(self, page_str, verbose=False):
        raise NotImplemented



class GroupleVolumesParser(IParser):
    def __init__(self):
        self.stop_symbs = ['\n', '  ']

    def parse(self, page_str, verbose=False):
        assert type(page_str) == str
        for symb in self.stop_symbs:
            page_str = page_str.replace(symb, '')

        etree_obj = html.fromstring(page_str)
        etree_obj = etree_obj.xpath("//div[@id='twitts']")[0]
        data = Volume(etree_obj, page_str)

        page_type = self._get_page_status(page_str)
        return {'comments': data.get_comments(),
                'links': data.get_links(),
                'ok_parsed': page_type == PageTypes.CORRECT,
                'status': page_type.name}
    
    def _get_page_status(self, page_str):
        classifier = GroupleVolumesClassifier(page_str)
        return classifier.get_type()
