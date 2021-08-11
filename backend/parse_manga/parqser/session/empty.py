from .base import BaseSession


class EmptySession(BaseSession):
    def auth(self, *args):
        pass
