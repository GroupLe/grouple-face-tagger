def self_path(el):
    return el.getroottree().getpath(el)


class Volumes():
    def __init__(self, etree_obj):
        self.volumes = []
        volumes = etree_obj.xpath(self_path(etree_obj) + "/div[contains(@class, 'chapters-link')]")[0]
        volumes = volumes.xpath(self_path(volumes) + '/table/tr')
        for vol in volumes:
            vol = vol.xpath(self_path(vol) + '/td/a')[0]
            self.volumes.append(vol.attrib['href'])

    def get_links(self):
        return self.volumes