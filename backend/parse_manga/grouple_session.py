from typing import Dict, List
from parqser.session import BaseSession


def read_spaceproxy_file(path: str) -> List[Dict[str, str]]:
    """Reader for spaceproxy.net proxy file"""
    proxies = []
    with open(path, 'r') as fin:
        for line in fin:
            values = line.strip().split(':')
            fields = ['address', 'port', 'login', 'password']
            proxy = {key: val for key, val in zip(fields, values)}
            proxies.append(proxy)

    # convert to requests module format
    requests_proxies = []
    for proxy_data in proxies:
        auth = proxy_data['login'] + ':' + proxy_data['password']
        proxy = auth + '@' + proxy_data['address'] + ':' + proxy_data['port']
        proxy = {'http': 'http://%s/' % proxy,
                 'https': 'https://%s/' % proxy}
        requests_proxies.append(proxy)
    return requests_proxies


class GroupleSession(BaseSession):
    """Example session class. Allows creating session for site grouple.co"""
    def __init__(self, proxy=None):
        super().__init__()
        self.proxy = proxy

    def auth(self, login: str, password: str):
        url = 'https://grouple.co/login/authenticate'
        payload = {'username': login, 'password': password}
        self._session.post(url, data=payload, proxies=self.proxy)
