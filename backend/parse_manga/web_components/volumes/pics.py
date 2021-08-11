from typing import List
from parqser.web_component import BaseComponent


class Pics(BaseComponent):

    @staticmethod
    def _get_links_substr(page_str: str) -> str:
        start = page_str.index('rm_h.init')
        start = page_str[start:].index('[') + start + 1
        stop = start
        brackets = 1
        while brackets != 0:
            if page_str[stop] == ']':
                brackets -= 1
            elif page_str[stop] == '[':
                brackets += 1
            stop += 1
        return page_str[start:stop - 1]

    @staticmethod
    def _make_links(links_substr: str) -> List[str]:
        links = eval(links_substr)
        links = list(map(lambda lst: lst[0] + lst[2], links))
        return links

    def parse(self, page_str: str) -> List[str]:
        s = self._get_links_substr(page_str)
        links = self._make_links(s)
        return links
