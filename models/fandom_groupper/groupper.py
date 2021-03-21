import pickle
from tqdm import tqdm
from entities import Title
from fandom import FandomGroup


class PersonIntersectionGroupper:
    def __init__(self, df=None, path=None):
        if df is not None:
            self.groups = self._calc_groups(df)
        elif path is not None:
            self.groups = pickle.load(open(path, 'rb'))
        else:
            raise AttributeError('One of arguments should be passed')

    def get_groups(self):
        return self.groups

    def save(self, path):
        pickle.dump(self.groups, open(path, 'wb'))

    def search_fandoms(self, person):
        candidates = []

        for group in self.groups:
            for title, persons in zip(group.titles, group.persons):
                for p in persons:
                    if person.lower() in p.lower():
                        candidates.append(group)
                        break
                else:
                    continue
                break
        return candidates

    def _calc_groups(self, df):
        # df - title to person dataframe
        titles, persons = self._get_titles_and_persons(df)
        titles = list(map(Title, titles))
        groups = self._group(titles, persons)
        return groups

    def _get_titles_and_persons(self, df):
        groupped = df.groupby('title')['name'].apply(list)
        groupped = list(groupped.items())
        titles, persons = zip(*groupped)
        return titles, persons

    def _group(self, titles, persons):
        groups = []

        initial_group = FandomGroup()
        initial_group.add(titles[0], persons[0])
        groups.append(initial_group)

        pbar = tqdm(total=len(titles))
        prev = 0
        while any(list(map(lambda t: not t.is_groupped, titles))):

            is_groupped_n = sum(list(map(lambda t: t.is_groupped, titles)))
            # print("Groupping progress: [%3d/%3d]" % (is_groupped_n, len(titles)))
            pbar.update(is_groupped_n - prev)
            prev = is_groupped_n

            for group in groups:
                # speeding up
                if not group.upd_manager.should_check():
                    continue
                else:
                    group.upd_manager.check()

                for i, title in enumerate(titles):
                    # is already in group
                    if title.is_groupped:
                        continue
                    # persons are similar to persons in group
                    if group.is_belong_by_persons(persons[i]):
                        group.add(title, persons[i])
                        group.upd_manager.update()
                        title.is_groupped = True

            # no items that most probably belongs to any existed group
            # Create new group of first ungroupped item
            for i, title in enumerate(titles):
                if not title.is_groupped:
                    new_group = FandomGroup()
                    new_group.add(title, persons[i])
                    groups.append(new_group)
                    title.is_groupped = True
                    break
        pbar.close()
        return groups
