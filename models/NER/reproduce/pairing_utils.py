import re

def get_pairing(s):
    # Fetches pairing in semi-conventional format from given string
    # Returns pair of characters or None
    separators = 'xх\\\/'
    regexp = r'\ [А-ЯA-Z][а-яА-Яa-zA-Z]*\ ?(x|х|Х|X|\\|\/)\ ?[А-ЯA-Z][а-яА-Яa-zA-Z]*'

    def hotfix(s):
        nexlines = 'Читать Страниц Приквел Продолжение'.split()
        for nl in nexlines:
            s = s.replace(nl, '')
        return s

    try:
        pairing = re.search(regexp, s).group(0)
        probable_pairing = None
        
        for sep in list(separators):
            if len(pairing.split(sep)) == 2:
                p1, p2 = pairing.split(sep)
                p1, p2 = map(str.strip, [p1, p2])
                if p1[0].isupper() and p2[0].isupper():
                    p1, p2 = hotfix(p1), hotfix(p2)
                    probable_pairing = (p1, p2)
                
        return probable_pairing
    except:
        return None
    
    
def get_probable_fandoms(df, p1, p2):
    # Searches fandom that contains both p1 and p2 characters' names. Name checking is fuzzy
    # Returns list of fandoms names
    p1_fandoms = df[df.name.apply(lambda s: any(map(lambda ss: ss==p1, s.split())))].title.tolist()
    p2_fandoms = df[df.name.apply(lambda s: any(map(lambda ss: ss==p2, s.split())))].title.tolist()
    fandoms = set(p1_fandoms).intersection(set(p2_fandoms))
    return list(fandoms)


def group_fandoms(arr):
    # Takes list of fandoms names. Returns dict of {general fandom name : belong fandoms names}
    def is_prefix(s, prefix):
        # smooth checking for containing prefix
        clear = lambda s: ''.join(list(filter(lambda c: c.isalpha() or c == ' ', s)))
        s = clear(s)
        prefix = clear(prefix)
        return s.startswith(prefix)
    
    prefixes = []
    groups = {}
    groupped = [False for _ in arr]
    arr = sorted(arr, key=lambda s: len(s)) # short common name is better then long
    
    for j, name in enumerate(arr):
        if groupped[j]:
            continue
        
        tokens = name.split()
        group_sizes = []
        ungroupped = [arr[i] for i in range(len(arr)) if not groupped[i]] # search matches only in unvisited titles
        # search the most common prefix
        for i in range(len(tokens)):
            prefix = ' '.join(tokens[:i+1])
            group_size = sum([is_prefix(title, prefix) for title in ungroupped])
            group_sizes.append(group_size)
            
            # perspectiveless group
            if group_size == 1:
                break
            
        
        if group_sizes[0] >= 7:
            # very big fandom group
            i = 1
            while i<len(tokens) and group_sizes[i] >= 5:
                i+=1
            prefix = ' '.join(tokens[:i])
            group = []
            for i, title in enumerate(ungroupped):
                if is_prefix(title, prefix):
                    group.append(title)
                    groupped[i] = True
            groups[prefix] = group

        elif group_sizes[0] >= 2:
            # big fandom group
            i = 1
            while i<len(tokens) and group_sizes[i] >= 2:
                i+=1
            prefix = ' '.join(tokens[:i])
            group = []
            for i, title in enumerate(arr):
                if not groupped[i] and is_prefix(title, prefix):
                    group.append(title)
                    groupped[i] = True
            groups[prefix] = group
    
    # add other ungroupped titles as independed groups
    for j, name in enumerate(arr):
        if not groupped[j]:
            groups[name] = [name]
            
    # Merge names witha and without ":"
    for key, vals in groups.items():
        if key.endswith(':'):
            if key[:-1] in groups.keys():
                groups[key[:-1]] += vals
                groups[key] = []
    for key in list(groups.keys()):
        if groups[key] == []:
            groups.pop(key)
        
    return groups