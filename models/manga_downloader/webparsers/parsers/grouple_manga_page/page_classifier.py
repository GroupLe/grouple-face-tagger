from .page_types import PageTypes


class GroupleMangaPageClassifier:
    '''
    Example of banning page classifier
    '''
    def __init__(self, page_str):
        self.page = page_str.lower()

    def _is_404(self):
        marker1 = 'Ой, ой, страничка потерялась'
        marker2 = 'Спокойно! Логи записаны. Все будет исправлено.'
        return marker1 in self.page or marker2 in self.page

    def _is_alive(self):
        marker1 = 'читать мангу с персой главы'
        return marker1 in self.page

    def get_type(self):
        if self._is_404():
            return PageTypes.E_404
        elif self._is_alive():
            return PageTypes.CORRECT
        else:
            return PageTypes.UNDEFINED
