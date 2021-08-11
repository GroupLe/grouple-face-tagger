from typing import List
from lxml import html
from parqser.web_component import BaseComponent


class Reviews(BaseComponent):
    def parse(self, source: str) -> List[str]:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]

        path = "/div[@class='posts-container']/div"
        items = etree.xpath(self.xpath(etree) + path)

        reviews = []
        for review in items:
            review = review.xpath(self.xpath(review))[0]
            reviews.append(review.text_content().strip())
        return reviews