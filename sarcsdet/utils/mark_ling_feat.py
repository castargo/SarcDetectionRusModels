from sarcsdet.configs.ling_feat_config import interjections, funny_marks


def funny_marks_feature(text):
    text_set = (str(text)).split()
    return sum([text_set.count(x) for x in funny_marks])


def interjections_feature(text):
    text_set = (str(text)).split()
    return sum([text_set.count(x) for x in interjections])


def get_popular_items(column):
    items = dict()
    for item_list in column:
        if item_list:
            for item in item_list:
                if item in items.keys():
                    items[item] += 1
                else:
                    items[item] = 1

    top_items = dict(sorted(items.items(), key=lambda item: item[1], reverse=True))

    for i, key in enumerate(top_items):
        top_items[key] = i + 1

    result = []
    for item_list in column:
        if item_list:
            result.append(min([top_items[item] for item in item_list]))
        else:
            result.append(0)
    
    return result
