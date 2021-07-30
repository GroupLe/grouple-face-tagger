import pandas as pd
from pathlib import Path
from typing import List
from nltk.stem.snowball import SnowballStemmer


class Character:
    def __init__(self, name: str, pic_link: str, link: str):
        self.name = name
        self.pic_link = pic_link
        self.link = link


class CandidatesSearch:
    def __init__(self, path: Path):
        self.df = self.get_df(path)
        self.stemmer = SnowballStemmer(language='russian')

    def get_df(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(Path(path), sep='\t')
        index_ = ['page_link', 'name', 'link', 'img_addr']
        df = df[index_]
        return df

    def get_characters_by_name(self, name: str) -> List[Character]:
        df_full = self.df[self.df['name'] == name]

        if df_full.empty:
            name_stemm = self.stemmer.stem(name).capitalize()
            df_stemm = (self.df[self.df['name'].str.contains(name_stemm)])
            df_stemm = df_stemm[df_stemm['name'].str.startswith(name_stemm)]
            df_chars = df_stemm.drop_duplicates(subset='link')
        else:
            df_chars = df_full.drop_duplicates(subset='link')

        characters = []
        for _, (page_link, name, link, img_addr) in df_chars.iterrows():
            char = Character(name, page_link[:20] + img_addr, page_link[:20] + link)
            characters.append(char)
        return characters

    def get_characters(self, names: List[str]) -> List[List[Character]]:
        characters = []
        for name in names:
            characters += self.get_characters_by_name(name)
        return characters
