def self_path(el):
    return el.getroottree().getpath(el)


class Header():
    def __init__(self, etree_obj):
        self.manga_type = etree_obj.xpath(self_path(etree_obj) + "/h1[@class='names']")[0]
        self.manga_type = self.manga_type.text.strip()
        self.title = etree_obj.xpath(self_path(etree_obj) + "/h1[@class='names']/span")[0].text

    def get_manga_type(self):
        return self.manga_type

    def get_title(self):
        return self.title