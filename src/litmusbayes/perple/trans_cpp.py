import json
import re

from src.litmusbayes import config
from src.litmusbayes.perple.perple import perpLE
from src.litmusbayes.utils.file_util import search_file, read_file

print_flag = True

reverse_func = '''
#define REVERSE_CTX(arr, n) do {      \
    int _i = 0, _j = (n) - 1;         \
    while (_i < _j) {                 \
        __typeof__(arr[0]) tmp = arr[_i]; \
        arr[_i] = arr[_j];            \
        arr[_j] = tmp;                \
        _i++;                         \
        _j--;                         \
    }                                 \
} while (0)

'''

perple_path = '/home/whq/Desktop/code_list/perple_test/perple_json'

def format_indent(code: str, indent_size: int = 2) -> str:
    """简单的缩进美化器，保证 for/if/asm 内的层次清晰"""
    lines = code.splitlines()
    formatted = []
    indent = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append("")
            continue

        # 遇到闭合大括号 } 缩进先减少
        if stripped.startswith("}"):
            indent -= 1

        formatted.append(" " * (indent * indent_size) + stripped)

        # 遇到以 { 结尾的行，缩进加一
        if stripped.endswith("{"):
            indent += 1

    return "\n".join(formatted)


def filter_Thread(input_file, output_file, litmus_file_path, perp_dict = None, thread_num = 2, mode = 'e'):

    litmus_name = litmus_file_path.split("/")[-1][:-7]
    with open(input_file, 'r') as f:
        cpp_code = f.read()
    litmus_content = read_file(litmus_file_path)
    # 0. get asm and get perpLE
    asm_list = []
    lines = litmus_content.splitlines()
    for line in lines:
        # 查找包含 P0 | P1 | ... 的行
        # 通常格式是 “ P0 | P1 | P2 ; ”
        if "P0" in line and "|" in line:
            # 统计一行里 | 的个数，线程数 = | 个数 + 1
            thread_num = line.count("|") + 1
            break
    for thread_index in range(thread_num):
        # Match the function body of static void *P0 (non-greedy match of the entire braces)
        func_pattern = re.compile(
            rf'(static\s+void\s*\*\s*P{thread_index}\s*\([^)]*\)\s*{{)(.*?)(^\}})',
            re.DOTALL | re.MULTILINE
        )
        func_match = func_pattern.search(cpp_code)

        if func_match:
            func_start = func_match.group(1)  # "static void *P0(...) {"
            func_body = func_match.group(2)  # Function internal code
            func_end = func_match.group(3)  # Closing brace }

            # Replace the original asm using regex
            asm_pattern = re.compile(r'asm\s+__volatile__\s*\(.*?\);\s*', re.DOTALL)
            asm_matches = asm_pattern.findall(func_body)
            for i, asm_code in enumerate(asm_matches):
                asm_list.append((thread_index, asm_code))

    if print_flag:
        for asm_code in asm_list:
            print(asm_code)
    if perp_dict is not None:
        trans_dict = perp_dict
    else:
        # raise ValueError("has not perp_dict")
        trans_dict = perpLE(litmus_content, asm_list, mode=mode)
    # if (mode == "h"):
    #     try:
    #         with open(f"{perple_path}/{litmus_name}.jsonl", 'w', encoding='utf-8') as jf:
    #             # ensure_ascii=False 允许写入中文等非ASCII字符（虽然这里主要是代码）
    #             # indent=4 会格式化输出，方便你打开文件查看代码结构
    #             json.dump(trans_dict, jf, ensure_ascii=False, indent=4)
    #         print(f"保存成功：{perple_path}/{litmus_name}.jsonl")
    #     except IOError as e:
    #         print(f"保存失败：{e}")
    #     print("trans_dict", trans_dict)

    # 1. update all P-thread functions
    for thread_index in range(thread_num):
        func_pattern = re.compile(
            rf'(static\s+void\s*\*\s*P{thread_index}\s*\([^)]*\)\s*{{)(.*?)(^\}})',
            re.DOTALL | re.MULTILINE
        )
        func_match = func_pattern.search(cpp_code)

        if func_match:
            func_start = func_match.group(1)  # "static void *P0(...) {"
            func_body = func_match.group(2)  # Function internal code
            func_end = func_match.group(3)  # Closing brace }

            # New asm code
            new_asm_code = trans_dict[f'P{thread_index}']
            new_asm_code += trans_dict[f'P{thread_index}_after_asm']
            new_asm_code += "mbar();"
            # print('new asm code:')
            # print(new_asm_code)

            # 找到 asm __volatile__ 的起始和结束
            start_idx = func_body.find("asm __volatile__ (")
            if start_idx == -1:
                raise ValueError("没找到 asm __volatile__ 起始位置")

            end_idx = func_body.find(");", start_idx)
            if end_idx == -1:
                raise ValueError("没找到 asm __volatile__ 结束位置")

            # 拼接新的函数体
            new_func_body = func_body[:start_idx] + new_asm_code + func_body[end_idx + 2:]

            trashed_array = trans_dict[f'P{thread_index}_var']
            for var in trashed_array:
                new_func_body = re.sub(rf'\s*int\s+{var}\s*;\s*\n', '', new_func_body)

            def insert_initializer(match):
                block_start = match.group(1)
                inits = trans_dict[f'P{thread_index}_init']
                return block_start + inits

            insert_pattern = re.compile(r'(for\s*\(int\s+_j\s*=\s*_stride\s*;.*?{\s*)')
            new_func_body = insert_pattern.sub(insert_initializer, new_func_body)


            func_code = func_start + new_func_body + func_end
            func_code = format_indent(func_code)
            # print('-----func_code-----')
            # print(func_code)

            func_start_idx = func_match.start()
            func_end_idx = func_match.end()
            cpp_code = cpp_code[:func_start_idx] + func_code + cpp_code[func_end_idx:]

            # print('new cpp_code:')
            # print(cpp_code)

    # 2. update statistics
    def find_matching_brace(code, start_index):
        """
        Starting from start_index, find the index of the } that matches the first {
        Returns the position of the closing brace
        """
        assert code[start_index] == '{'
        stack = 1
        for i in range(start_index + 1, len(code)):
            if code[i] == '{':
                stack += 1
            elif code[i] == '}':
                stack -= 1
                if stack == 0:
                    return i
        raise ValueError("Matching } not found")

    # 2.1 Find the start of the zyva function
    func_match = re.search(r'static\s+void\s*\*\s*zyva\s*\([^)]*\)\s*{', cpp_code)
    if not func_match:
        print("Function zyva not found")
        exit()

    func_start_idx = func_match.start()
    func_body_start = func_match.end() - 1  # Position of the first {

    # 2.2 Find the end of the function
    func_body_end = find_matching_brace(cpp_code, func_body_start)
    func_body = cpp_code[func_body_start + 1: func_body_end]  # Exclude the outer braces

    # 2.3 Find the n_run for loop
    for_match = re.search(r'for\s*\(\s*int\s+n_run\s*=.*?\)\s*{', func_body)
    if not for_match:
        print("n_run loop not found")
        exit()

    for_start_idx = for_match.start()
    for_body_start = for_match.end() - 1  # Position of the {

    # 2.4 Find the end of the loop body
    for_body_end = find_matching_brace(func_body, for_body_start)
    for_body = func_body[for_body_start + 1: for_body_end]


    # 2.5 Insert printf at the end of the loop body
    for_body_modified = for_body + f'\n{trans_dict['exhaust']};\n'

    # 2.6 Reconstruct the modified for loop
    new_for_loop = func_body[for_start_idx:for_body_start + 1] + for_body_modified + func_body[for_body_end]

    # 2.7 Reconstruct the modified function body
    new_func_body = func_body[:for_start_idx] + new_for_loop + func_body[for_body_end + 1:]
    # print("--------new_func_body--------")
    # print(new_func_body)

    # 2.8 Reconstruct the entire function
    new_func = cpp_code[func_start_idx: func_body_start + 1] + new_func_body + cpp_code[func_body_end]
    # print("--------new_func--------")
    # print(new_func)
    # Replace the original function in the cpp code
    cpp_code = cpp_code[:func_start_idx] + reverse_func + new_func + cpp_code[func_body_end + 1:]

    # Write the modified code to a new cpp file
    with open(output_file, "w") as f:
        # print("cppcode")
        cpp_code = cpp_code.replace(
            "add_outcome(hist,1,o,cond);",
            "/* add_outcome(hist,1,o,cond); */"
        )
        cpp_code = re.sub(
            r'^[ \t]*(fatal\(.*?\))[^\n]*$',
            r'/* \1 */',
            cpp_code,
            flags=re.MULTILINE
        )
        f.write(cpp_code)

    print(f"New file generated: {output_file}")




if __name__ == "__main__":

    litmus_path_dir = [
        # ("PPOCA.c", "PPOCA_change.c", f'{config.TEST_DIR}/experiment/exp1_litmus/PPOCA.litmus'),
        # ("R.c", "R_change.c","R_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/R.litmus'),
        # ("SB.c", "SB_change.c", "SB_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/SB.litmus'),
        # ("/home/whq/Desktop/code_list/perple_test/all_litmus/LB+fence.i+ctrlfencei/LB+fence.i+ctrlfencei.c", "LB+fence.i+ctrlfencei_change.c", "LB+fence.i+ctrlfencei_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/LB+fence.i+ctrlfencei.litmus'),
        # ("MP.c", "MP_change.c", "MP_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/MP.litmus'),
        # ("S.c", "S_change.c", "S_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/S.litmus'),
        # ("CoRR2-cleaninit.c", "CoRR2-cleaninit.c", "CoRR2-cleaninit.c", f'/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/CoRR2-cleaninit.litmus'),
        # ("/home/whq/Desktop/code_list/perple_test/all_litmus/IRIW+addr+ctrlfencei/IRIW+addr+ctrlfencei.c", "IRIW+addr+ctrlfencei_change.c", "IRIW+addr+ctrlfencei_change_h.c", f'/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/IRIW+addr+ctrlfencei.litmus'),
        # ("/home/whq/Desktop/code_list/perple_test/all_litmus/ISA2+fence.rw.rw+addr+ctrlfencei/ISA2+fence.rw.rw+addr+ctrlfencei.c", "ISA2+fence.rw.rw+addr+ctrlfencei_change.c", "ISA2+fence.rw.rw+addr+ctrlfencei_change_h.c", f'/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/ISA2+fence.rw.rw+addr+ctrlfencei.litmus'),
        # ("/home/whq/Desktop/code_list/perple_test/all_litmus/ISA02/ISA02.c", "ISA02_change.c", "ISA02_change_h.c", f'/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/ISA02.litmus'),
        # ("/home/whq/Desktop/code_list/perple_test/all_litmus/ISA02/ISA02.c", "ISA02_change.c", "ISA02_change_h.c", f'/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/ISA02.litmus'),
        # ("/home/whq/Desktop/code_list/perple_test/all_litmus/MP/MP.c",
        #  "MP_change.c", "MP_change_h.c",
        #  f'/home/whq/Desktop/code_list/perple_test/litmus_test/MP.litmus'),
        ("/home/whq/Desktop/code_list/perple_test/all_litmus/LB+po+poaqp+NEW/LB+po+poaqp+NEW.c",
         "LB+po+poaqp+NEW_change.c", "LB+po+poaqp+NEW_change_h.c",
         f'/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/LB+po+poaqp+NEW.litmus'),
    ]

    for input_path, output_path_e, output_path_h, litmus_file_path in litmus_path_dir:
        litmus_content = read_file(litmus_file_path)
        filter_Thread(input_path, output_path_e, litmus_file_path, thread_num=2)

        filter_Thread(input_path, output_path_h, litmus_file_path, thread_num=2, mode='h')




