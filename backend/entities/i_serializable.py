from typing import Dict


class ISerializable:
    def __init__(self, *args, **kwargs):
        raise NotImplemented

    def to_json(self) -> Dict:
        raise NotImplemented