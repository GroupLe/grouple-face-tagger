from typing import List
from lxml import html
from parqser.web_component import BaseComponent


class ChaptersComponent(BaseComponent):
    def parse(self, source: str) -> List[str]:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]

        path = "/div[@class='expandable chapters-link']/table[@class='table table-hover']/tr"
        items = etree.xpath(self.xpath(etree) + path)

        parts = []
        for part in items:
            part = part.xpath(self.xpath(part) + "/td/a")[0]
            parts.append(part.get("href"))
        return parts