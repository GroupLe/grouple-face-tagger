from typing import List
from lxml import html
from grouple.backend.parse_manga.parqser.web_component import BaseComponent


class Comments(BaseComponent):
    def parse(self, source: str) -> List[str]:
        etree = html.fromstring(source).xpath("//div[@class='leftContent']")[0]

        path = "/div[@class='posts-container expandable post-container_ forum-topic']/div"
        items = etree.xpath(self.xpath(etree) + path)

        comments = []
        for comm in items:
            comm = comm.xpath(self.xpath(comm) + "/div[@class='media-body']")[0]
            comments.append(comm.text_content().strip())
        return comments
