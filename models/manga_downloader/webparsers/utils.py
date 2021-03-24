def load_proxies():
    # loads proxy addresses data in format {address: str, port: str, login: str, password: str}
    proxies = []
    with open('proxylist.txt', 'r') as fin:
        for line in fin:
            values = line.strip().split(':')
            fields = 'address port login password'.split()
            proxy = {key: val for key, val in zip(fields, values)}
            proxies.append(proxy)
    return proxies