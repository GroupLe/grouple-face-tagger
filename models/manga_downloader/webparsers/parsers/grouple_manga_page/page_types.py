from enum import Enum

class PageTypes(Enum):
    CORRECT = 1
    E_404 = 2
    IP_BANNED = 3
    USER_BANNED = 4
    UNDEFINED = 5
    ERR = 6
