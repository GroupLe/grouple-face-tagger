import requests
from .session import ISession



class EmptySession(ISession):
    def __init__(self, proxy_data=None):
        '''
        @param proxy_data:  dict of login, password, address and port for proxy
        '''
        self.proxy = None # default is no proxy
        
        if proxy_data:
            auth = proxy_data['login'] + ':' + proxy_data['password']
            proxy = auth + '@' + proxy_data['address'] + ':' + proxy_data['port']
            self.proxy = {'http': 'http://%s/' % proxy,
                          'https':'https://%s/' % proxy}        
    
    def load_page(self, url):
        res = requests.get(url, proxies=self.proxy)
        return res