from typing import List
from lxml import html
from parqser.web_component import BaseComponent


class CommentsPage(BaseComponent):
    def parse(self, source: str) -> List[str]:
        etree = html.fromstring(source).xpath("//div[@class='pageBlock container']/div[@class='twitter twitts-src']")[0]

        path = "//div[@class='mess']"
        items = etree.xpath(self.xpath(etree) + path)

        comms = []
        for comm in items:
            review = comm.xpath(self.xpath(comm))[0]
            comms.append(review.text_content().strip())
        return comms
