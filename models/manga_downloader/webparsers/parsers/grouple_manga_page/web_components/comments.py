def self_path(el):
    return el.getroottree().getpath(el)


class Comments():
    def __init__(self, etree_obj):
        self.comments = []
        items = etree_obj.xpath(self_path(etree_obj) + "/div[@class='posts-container expandable post-container_ forum-topic']/div")
        for i, comm in enumerate(items):
            comm = comm.xpath(self_path(comm) + "/div[@class='media-body']")[0]
            self.comments.append(comm.text_content().strip())

    def get_text(self):
        return self.comments