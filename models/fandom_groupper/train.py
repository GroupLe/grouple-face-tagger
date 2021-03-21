import pandas as pd
from groupper import PersonIntersectionGroupper


def load_data(path):
    ydf1 = pd.read_csv(root + 'you_anime_characters_refs.csv', sep='\t')
    ydf1['anime_url'] = ydf1.page_link.apply(lambda s: s.split('/')[-1])

    ydf2 = pd.read_csv(root + 'you_anime_refs.csv', sep='\t')
    ydf2['anime_url'] = ydf2.link.apply(lambda s: s.split('/')[-1])

    ydf = pd.merge(ydf1, ydf2, on='anime_url')
    ydf = ydf['title name'.split()]

    ydf.title = ydf.title.apply(str.strip)
    ydf.name = ydf.name.apply(str.strip)

    ydf = ydf.sort_values(by='title')
    return ydf


def select_popular(groups):
    names = ['Наруто', 'Блич', 'Ван-Пис', 'Атака', 'Баскетбол', 'Бездомный', 'Волейбол', 'Гинтама', 'Граница']
    pure_groups = []
    for g in groups:
        for title in g.titles:
            for name in names:
                if name in title.orig_name:
                    pure_groups.append(g)
                    break
            else:
                continue
            break
    return pure_groups


if __name__ == '__main__':
    root = '../characters_parser/data/'
    ydf = load_data(root)
    groupper = PersonIntersectionGroupper(df=ydf)
    print('Number of created groups:', len(groupper.groups))

    groupper.save('./groupper_model.pkl')
    print('Model saved')

    # groupper = PersonIntersectionGroupper(path='./groupper_model.pkl')

    print('=====================================')
    print('Example with popular titles:')
    pop_gr = select_popular(groupper.groups)
    print('\n\n'.join(list(map(str, pop_gr))))


    print('\n\n\n\n=====================================')
    print("Example with first n titles:")
    print('\n\n'.join(list(map(str, groupper.groups[:100]))))






