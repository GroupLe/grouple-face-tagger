from lxml import html
from parqser.web_component import BaseComponent


class Title(BaseComponent):
    def parse(self, source: str) -> str:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]
        title = etree.xpath(self.xpath(etree) + "/h1[@class='names']/span")[0].text
        return title


class MangaType(BaseComponent):
    def parse(self, source: str) -> str:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]
        manga_type = etree.xpath(self.xpath(etree) + "/h1[@class='names']")[0]
        manga_type = manga_type.text.strip()
        return manga_type
