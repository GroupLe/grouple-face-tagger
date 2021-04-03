

def read_conll(path):
    """
    Returns list of sentences. Sentence is a list of pairs (word, NE-label)
    """
    data = open(path).read().split('\n\n')[1:]
    data = list(map(str.splitlines, data))
    get_token_and_ne = lambda s: (s.split()[0], s.split()[-1])
    data = list(map(lambda sent: list(map(get_token_and_ne, sent)), data))
    return data
