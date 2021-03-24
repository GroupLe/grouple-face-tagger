def self_path(el):
    return el.getroottree().getpath(el)


class Volume():
    def __init__(self, etree_obj, page_str):
        try:
            self.comments = etree_obj.xpath(self_path(etree_obj) + "/div/div/div[@class='mess']")
            self.comments = list(map(lambda comm: comm.text_content(), self.comments))
        except:
            self.comments = []
        self.links = self._get_links(page_str)
        
    def get_comments(self):
        return self.comments
    
    def get_links(self):
        return self.links 
    
    def _get_links(self, page_str):
        def get_links_substr(page_str):
            start = page_str.index('rm_h.init')
            start = page_str[start:].index('[') + start + 1
            stop = start
            brackets = 1
            while brackets != 0:
                if page_str[stop] == ']':
                    brackets -= 1
                elif page_str[stop] == '[':
                    brackets += 1
                stop += 1
            return page_str[start:stop-1]

        def make_links(links_substr):
            links = eval(links_substr)
            links = list(map(lambda lst: lst[0] + lst[2], links))
            return links

        s = get_links_substr(page_str)
        links = make_links(s)
        return links