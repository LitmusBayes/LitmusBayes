
import re


def findone(compile_string, dst_string, flags=""):
    matches = findall(compile_string, dst_string, flags=flags)
    assert len(matches) <= 1, f"matches (len: {len(matches)}): {matches}, compile_string: {compile_string}"
    if len(matches) == 0:
        return ""
    else:
        return matches[0]


def findall(compile_string, dst_string, flags=""):
    """
    example:
        litmus_strs = regex_util.findall(r"% Results for .*? %\n.*?Time .*?\n", log_content, re.S)

    :param compile_string:
    :param dst_string:
    :param flags:
    :return:
    """
    if flags == "":
        pattern = re.compile(compile_string)
    else:
        pattern = re.compile(compile_string, flags)
    matches = re.findall(pattern, dst_string)
    return matches


def sub(compile_string, replaced_string, dst_string, flags=""):
    if flags == "":
        pattern = re.compile(compile_string)
    else:
        pattern = re.compile(compile_string, flags)
    string_after = re.sub(pattern, replaced_string, dst_string)
    return string_after


def split_spaces(cmd):
    """
    split multiple whitespaces
    refer to: https://theprogrammingexpert.com/python-replace-multiple-spaces-with-one-space/
    """
    splits = cmd.split()
    return splits
