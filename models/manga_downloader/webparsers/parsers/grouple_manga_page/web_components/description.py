def self_path(el):
    return el.getroottree().getpath(el)


class Description():
    def __init__(self, etree_obj):
        self.text = etree_obj.xpath(self_path(etree_obj) + "/div[@class='expandable']/div")[1]
        self.text = self.text.text_content().strip()

    def get_text(self):
        return self.text