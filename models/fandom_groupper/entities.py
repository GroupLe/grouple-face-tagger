
class Title:
    def __init__(self, name):
        self.orig_name = name
        self.clear_name = self._prepare(name)
        self.is_groupped = False

    def _prepare(self, name):
        new_name = ''
        for i, c in enumerate(name):
            if c.isalpha() or c.isdigit() or c == ' ':
                new_name += c
            else:
                if len(new_name) > 0 and new_name[-1] != ' ':
                    new_name += ' '
        new_name = ' '.join(new_name.split())
        return new_name

    def is_prefix_with(self, prefix: str):
        tokens_orig = self.clear_name.split()
        tokens_pref = prefix.split()
        return tokens_orig == tokens_pref

    def get_clear_tokens(self):
        return self.clear_name.split()

    def get_orig_tokens(self):
        return self.orig_name

    def get_clear_name(self):
        return self.clear_name

    def get_orig_name(self):
        return self.orig_name
