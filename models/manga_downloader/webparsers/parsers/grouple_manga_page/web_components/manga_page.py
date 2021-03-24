from .header import Header
from .description import Description
from .comments import Comments
from .volumes import Volumes


class MangaPage:
    def __init__(self, etree_obj):
        self.header = Header(etree_obj)
        self.description = Description(etree_obj)
        self.comments = Comments(etree_obj)
        self.volumes = Volumes(etree_obj)

    def _clean(self, text):
        return str(text.replace('\t','').replace('\n','').replace('\r', ''))

    def get_manga_type(self):
        return self.header.get_manga_type()

    def get_title(self):
        text = self.header.get_title()
        text = self._clean(text)
        return text

    def get_description(self):
        text = self.description.get_text()
        text = self._clean(text)
        return text

    def get_comments(self):
        text = self.comments.get_text()
        return text

    def get_volumes(self):
        return self.volumes.get_links()