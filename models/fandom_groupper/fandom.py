from statistics import mean
from entities import Title


class GroupUpdateManager:
    def __init__(self):
        self.n_checks = 0
        self.checks_limit = 2

    def check(self):
        self.n_checks += 1

    def should_check(self):
        return self.n_checks < self.checks_limit

    def update(self):
        self.n_checks = 0


class FandomGroup:
    def __init__(self):
        self.upd_manager = GroupUpdateManager()
        self.titles = []
        self.persons = []
        self.all_persons = set()

    def _similarity(self, persons1, persons2):
        isec = len(set(persons1).intersection(set(persons2)))
        similarity = max(isec / len(persons1), isec / len(persons2))
        return similarity

    def add(self, title: Title, persons: list):
        self.titles.append(title)
        self.persons.append(persons)
        for person in persons:
            self.all_persons.add(person)

    def is_belong_by_persons(self, new_persons):
        sim_f = lambda core_persons: self._similarity(core_persons, new_persons)
        similarities = list(map(sim_f, self.persons))
        # Very similar to at least one title
        if mean(similarities) > 0.65:
            return True
        # Similar at all
        if self._similarity(self.all_persons, new_persons) > 0.9:
            return True
        return False

    def is_belong_by_title(self, title: Title):
        prefixes_l1 = set(list(map(lambda t: t.get_clear_tokens()[0], self.titles)))
        return title.get_clear_tokens()[0] in prefixes_l1

    def __repr__(self):
        per_fd = ['%s %d' % (title.orig_name, len(persons)) for title, persons in zip(self.titles, self.persons)]
        per_fd = '\n'.join(per_fd)
        return f'All persons n {len(self.all_persons)}\n{per_fd}'
