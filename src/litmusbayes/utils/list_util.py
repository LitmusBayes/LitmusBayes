

def remove_list(src_list, rm_list):
    return get_uniq_in_src(src_list, rm_list)


def remove_duplicates(list):
    new_list = []
    for node in list:
        if node not in new_list:
            new_list.append(node)
    # print(f"before: {len(list)}")
    list.clear()
    list.extend(new_list)
    # print(f"after: {len(list)}")


def has_duplicates(list):
    new_list = []
    for node in list:
        if node not in new_list:
            new_list.append(node)
    return len(new_list) != len(list)


def to_lower(list):
    new_list = []
    for node in list:
        new_list.append(node.lower())
    return new_list


def get_intersection(src_list, dst_list):
    intersection = []
    for src in src_list:
        if src in dst_list:
            intersection.append(src)
    return intersection


def is_intersect(src_list, dst_list):
    inter = get_intersection(src_list, dst_list)
    return len(inter) != 0


def is_not_intersect(src_list, dst_list):
    inter = get_intersection(src_list, dst_list)
    return len(inter) == 0


def is_same(src_list, dst_list):
    if is_subset(src_list, dst_list) and len(src_list) == len(dst_list):
        return True
    return False


def is_subset(src_list, dst_list):
    """
    check if src_list is a subset of dst_list
    """
    for src in src_list:
        if src not in dst_list:
            return False
    return True


def get_union(src_list, dst_list):
    union = []
    for src in src_list:
        union.append(src)
    for dst in dst_list:
        if dst not in union:
            union.append(dst)
    return union


def get_uniq_in_src(src_list, dst_list, is_print=False):
    uniq = []
    for src in src_list:
        if src not in dst_list:
            uniq.append(src)
    if is_print:
        print_list(uniq, "uniq")
        print(f"uniq len: {len(uniq)}")
    return uniq


def print_list(list, header="", cnt=-1):
    # print(f"\n{header} list info:")
    cur_cnt = 0
    for node in list:
        cur_cnt += 1
        if cnt < 0:
            print(f"{node}")
        elif cnt > 0 and cur_cnt < cnt:
            print(f"{node}")
    print('\n')


def to_string(list, strip=True):
    string = ""
    for node in list:
        if strip:
            string += f"{node.strip()}\n"
        else:
            string += f"{node}\n"
    return string.strip()


def remove_empty_strings(list):
    new_list = []
    for node in list:
        if len(node) != 0:
            new_list.append(node)
    return new_list


def merge_two_lists(a_list, b_list):
    merge_list = []
    merge_list.extend(a_list)
    merge_list.extend(b_list)
    return set(merge_list)
