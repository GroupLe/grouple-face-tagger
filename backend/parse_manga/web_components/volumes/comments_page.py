from typing import List
from lxml import html
import regex as re
from parqser.web_component import BaseComponent


class CommentsPage(BaseComponent):
    def parse(self, source: str) -> List[str]:

        comments = {}
        etree = html.fromstring(source)

        divs = etree.xpath("//div[contains(@class, 'hide')]")
        for div in divs:
            if len(list(div.classes)) > 1:
                class_name = str(list(div.classes)[1])
                if class_name.startswith('cm_'):
                    num = re.findall('(\d+)', class_name)
                    comms_div = div.xpath(self.xpath(div) + "//div[@class='mess']")
                    comms = []
                    for comm in comms_div:
                        # print(comm.text_content().strip())
                        comms.append(comm.text_content().strip())
                    comments[int(num[0])] = comms
        return comments
