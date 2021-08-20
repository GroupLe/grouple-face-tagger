from lxml import html
import sys
import requests as r
from typing import List, Dict
from parqser.web_component import BaseComponent


class CharactersComponent(BaseComponent):
    def parse(self, source: str) ->List[Dict[str, str]]:
        etree = html.fromstring(source)
        path = "//div[@class='inner']/a/img"
        items = etree.xpath(path)
        characters = []

        for i in items:
            url = 'https://you-anime.ru' + i.items()[0][1]
            name = i.items()[1][1]
            character = {name: url}
            characters.append(character)

        return characters


if __name__ == '__main__':
    characters = Characters()
    source = r.get('https://you-anime.ru/characters').text
    print(characters.parse(source))
