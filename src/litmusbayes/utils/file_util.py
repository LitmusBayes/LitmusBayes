# MIT License
#
# Copyright (c) 2023 DehengYang (dehengyang@qq.com)
#                    Xuezheng (xuezhengxu@126.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import os
import shutil

from src.litmusbayes.log import *
from . import dir_util


def _read_file_to_str(file_path):
    with io.open(file_path, encoding='utf-8', mode='r') as f:
        return f.read()


def _read_file_to_list_strip(file_path, skip_empty_line=True):
    assert os.path.exists(file_path)

    stripped_lines = []
    with io.open(file_path, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) == 0:
                logger.info("skip empty line.")
                continue
            stripped_lines.append(line.strip())
    return stripped_lines


def _read_file_to_list_no_strip(file_path):
    assert os.path.exists(file_path)
    ori_lines = []
    with io.open(file_path, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        for line in lines:
            ori_lines.append(line)
    return ori_lines


def search_files(name: str, path: str, suffix: str = ''):
    return [file for file in list_files(path) if file.endswith('/' + name + suffix)]


def search_file(name: str, path: str, suffix: str = ''):
    files = search_files(name, path, suffix)
    match len(files):
        case 0:
            # print(f'[ERROR] cannot find {name}{suffix} in {path}')
            return None
        case 1:
            return files[0]
        case _:
            # print(f'[Warning] multiple {name}.{suffix} found in {path}. {files}')
            return files[0]


def read_file(file_path: str, tolist=False, strip=False):
    if tolist:
        if strip:
            return _read_file_to_list_strip(file_path)
        else:
            return _read_file_to_list_no_strip(file_path)
    else:
        return _read_file_to_str(file_path)


def read_files(directory: str, suffix: str = '', tolist=False, strip=False):
    all_files = []
    for root, dirs, files in os.walk(directory):
        all_files.extend([os.path.join(root, file) for file in files if file.endswith(suffix)])

    return [(f, read_file(f, tolist=tolist, strip=strip)) for f in all_files]


def list_files(directory: str, suffix: str = ''):
    """
    :param directory: the target directory
    :param suffix: the suffix of target files
    :return: a list of filenames
    """
    all_files = []
    for root, dirs, files in os.walk(directory):
        all_files.extend([os.path.join(root, file) for file in files if file.endswith(suffix)])

    return all_files


MIN_DIR_LEN = 10


def get_file_name_from_path(file_path: str, with_suffix=False):
    """
    :param with_suffix:
    :param file_path:
    :return: file name without suffix

    e.g., a/b/c.litmus -> c
    """
    assert '.' in file_path and '/' in file_path, f"file_path: {file_path}"
    if with_suffix:
        return file_path.rsplit('/', 1)[-1]
    return file_path.rsplit('/', 1)[-1].rsplit(".", 1)[0]


def rm_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        logger.info(f'rm file: {file_path}')


def rm_all_content_in_dir_except(dir_path, except_file_name):
    logger.info(
        f"remove all content in {dir_path} except file: {except_file_name}")

    # to avoid delete important dir
    if len(dir_path) < MIN_DIR_LEN:
        assert False, "dangerous rm."

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name != except_file_name:
            os.remove(file_path)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


def remove_files(dir):
    rm_all_content_in_dir(dir)


def rm_all_content_in_dir(dir_path):
    INFO(f"remove all content in {dir_path}")

    # to avoid delete important dir
    if len(dir_path) < MIN_DIR_LEN:
        assert False, "dangerous rm."

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


def rm_files_with_suffix_in_dir(dir_path, suffix):
    # to avoid delete important dir
    if len(dir_path) < MIN_DIR_LEN:
        assert False, "dangerous rm."

    logger.info(f'rm files in dir: {dir_path}')
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith(suffix):
                os.remove(file_path)


def backup_dir(src_dir, dst_dir):
    assert os.path.isdir(src_dir)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def rm_dir(dir_path):
    if not os.path.exists(dir_path):
        return

    # to avoid delete important dir
    if len(dir_path) < MIN_DIR_LEN:
        assert False, "dangerous rm."

    if os.path.isdir(dir_path):
        logger.info(f"rm dir: {dir_path}")
        shutil.rmtree(dir_path)


def rm_file_safe_contain(file_path, contain_word):
    if contain_word not in file_path:
        assert False

    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        logger.info(f'rm file: {file_path}')


def rm_dir_safe_contain(dir_path, contain_word):
    if contain_word not in dir_path:
        assert False, "unsafe, exit"

    rm_dir(dir_path)


def rm_dir_safe_contain(dir_path, contain_word_list):
    for contain_word in contain_word_list:
        if contain_word not in dir_path:
            assert False, "unsafe, exit"

    rm_dir(dir_path)


def clear_file(file_path):
    """
    remove all content in the file
    """
    write_to_file(file_path, "", False)


def write_to_file(file_path, string, append=True):
    write_str_to_file(file_path, string, append=append)


def write_str_to_file(file_path, string, append=True):
    dir_util.mk_dir_from_file_path(file_path)

    mode = 'w'
    if append:
        mode = "a+"

    with open(file_path, mode) as f:
        f.write(string)


def write_line_to_file(file_path, line, line_break=True, append=True):
    dir_util.mk_dir_from_file_path(file_path)

    if line_break:
        line = line + "\n"

    mode = 'w'
    if append:
        mode = "a+"

    with open(file_path, mode) as f:
        f.write(line)


def write_list_to_file(file_path, lines_list, append=False, line_break=True):
    dir_util.mk_dir_from_file_path(file_path)

    mode = 'w'
    if append:
        mode = "a+"

    with open(file_path, mode) as f:
        for line in lines_list:
            if line_break:
                line = str(line) + "\n"
            f.write(line)

def get_file_name_without_suffix_by_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]