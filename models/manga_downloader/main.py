from downloader import Downloader

if __name__ == '__main__':
    d = Downloader()
    data = d.download('https://readmanga.live/put_samuraia')
    print('--')
    print(data)