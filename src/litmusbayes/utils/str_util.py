



def remove_suffix(old_str:str,suffix:str):
    return old_str[:len(old_str)-len(suffix)]


def remove_prefix(old_str:str,prefix:str):
    return old_str[len(prefix):]

def get_substring(old_str:str,suffix:str,prefix:str):
    return old_str[len(prefix):len(old_str)-len(suffix)]