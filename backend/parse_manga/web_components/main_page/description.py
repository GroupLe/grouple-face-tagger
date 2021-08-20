from lxml import html
from parqser.web_component import BaseComponent


class Description(BaseComponent):
    def parse(self, source: str) -> str:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]
        text = etree.xpath(self.xpath(etree) + "/div[@class='expandable']/div")[1]
        text = text.text_content().strip()
        return text
