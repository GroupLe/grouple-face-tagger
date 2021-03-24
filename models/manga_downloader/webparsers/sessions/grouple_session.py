import requests
from requests.auth import HTTPProxyAuth
from .session import ISession



class GroupleSession(ISession):
    '''
    Example session class. Allows creating session for site grouple.co
    '''
    def __init__(self, proxy_data=None):
        '''
        @param proxy_data:  dict of login, password, address and port for proxy
        '''
        self.session = requests.Session()
        self.session.trust_env = False
        self.proxy = None # default is no proxy
        
        if proxy_data:
            auth = proxy_data['login'] + ':' + proxy_data['password']
            proxy = auth + '@' + proxy_data['address'] + ':' + proxy_data['port']
            self.proxy = {'http': 'http://%s/' % proxy,
                          'https':'https://%s/' % proxy}
        
        url = 'https://grouple.co/login/authenticate'
        payload = {'username': 'YourName', 'password': 'YourPsw'}
        self.session.post(url, data=payload, proxies=self.proxy)
    
    
    def load_user(self, url):
        res = self.session.get(url, proxies=self.proxy)
        return res