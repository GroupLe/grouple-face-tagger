from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

class Comments:
    def __init__(self, comments):
        self.comments = list(map(self.clear, comments))
        self.labels = list(map(lambda s: ['O' for _ in s.split()], self.comments))

    
    def clear(self, text, stem=False):
        text = self.split_pairings(text)
        text = self.tokenize(text)
        text = self.del_spec_symbols(text)
        return text

    def split_pairings(self, text):
        s = text[0]
        for i, c in enumerate(text[1:], start=1):
            prev = text[i-1]
            if c.isalpha() and prev.isalpha():
                if c.isupper() and not prev.isupper():
                    s += ' '
            s += c
        return s
    
    def tokenize(self, text):
        text = tokenizer.tokenize(text)
        text = ' '.join(text)
        return text
    
    def del_spec_symbols(self, text):
        ranges = 'ая АЯ az AZ 09'.split()
        allowed = ''
        for cfrom, cto in ranges:
            chars = list(map(chr, range(ord(cfrom), ord(cto)+1))) # range of symbols
            allowed += ''.join(chars)
        punkt = ',.!? '
        
        s = text[0] if text[0] in allowed else ''
        for i, c in enumerate(text[1:], start=1):
            prev = text[i-1]
            if prev in punkt and c in punkt:
                # do not allow sequential punktuation such as ...
                continue
            if c in allowed or c in punkt:
                s += c
        return s.strip()
    
    
class Manga:
    def __init__(self, comments):
        self.comments = Comments(comments)