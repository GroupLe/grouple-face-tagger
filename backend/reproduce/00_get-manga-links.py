import pandas as pd
import regex as re
from typing import List, Tuple


def get_links(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    titles = []
    links = []
    for i in df.iterrows():
        cur_string = ''.join(i[1])
        flag = 0
        obj = []
        for j in cur_string:
            if j == '(':
                flag = 1
            if flag == 1:
                obj.append(j)
            if j == ')':
                flag = 0
                cur_string = ' '.join(obj)
                cur_string = cur_string.replace(" ", "")
                items = cur_string.split(',')
                if len(items) < 5:
                    continue
                for item in items:
                    if re.search('^(\'https:\/\/static.readmanga).*', item) != None:
                        title = items[4].replace("'", "")
                        titles.append(title)
                        link = 'https://readmanga.live/' + title
                        links.append(link)
                obj = []

    return titles, links

if __name__ == '__main__':

    df = pd.read_csv('../../data/backend/raw/elements.csv', sep='/t', engine ='python')

    titles, links = get_links(df)

    df = pd.DataFrame({'title': titles, 'link': links})
    df.to_csv('../../data/backend/raw/links.csv')
