from lxml import html
from grouple.backend.parse_manga.parqser.web_component import BaseComponent


class Title(BaseComponent):
    def parse(self, source: str) -> str:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]
        title = etree.xpath(self.xpath(etree) + "/h1[@class='names']/span")[0].text
        return title
