import json
import os
import re
from time import process_time_ns

from src.litmusbayes import config
from src.litmusbayes.analysis import RVWMO
from src.litmusbayes.bayes.util import get_files
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.prog import StoreInst, LoadInst, FenceInst, AmoInst, MoFlag, IType, FenceTsoInst, FenceIInst
from src.litmusbayes.utils.file_util import read_file


def get_vector_by_inst(inst):
    # vector = [R,W,amo,lr,sc,RL,AQ,Fence.rw.rw*,Fence.tso,Fence.i]
    vector = [0,0,0,0,0,0,0,0,0,0]
    if isinstance(inst, StoreInst):
        vector[1] = 1
    elif isinstance(inst, LoadInst):
        vector[0] = 1
    elif isinstance(inst, FenceInst):
        vector[7] = 1
    elif isinstance(inst, FenceTsoInst):
        vector[8] = 1
    elif isinstance(inst, FenceIInst):
        vector[9] = 1
    elif isinstance(inst, AmoInst):
        if inst.type == IType.Amo:
            vector[2] = 1
            if inst.flag == MoFlag.Strong:
                vector[5] = 1
                vector[6] = 1
            elif inst.flag == MoFlag.Acquire:
                vector[6] = 1
            elif inst.flag == MoFlag.Release:
                vector[5] = 1
        elif inst.type == IType.Lr or inst.type == IType.Sc:
            if inst.type == IType.Lr:
                vector[3] = 1
            elif inst.type == IType.Sc:
                vector[4] = 1
            if inst.flag == MoFlag.Strong:
                vector[5] = 1
                vector[6] = 1
            elif inst.flag == MoFlag.Acquire:
                vector[6] = 1
            elif inst.flag == MoFlag.Release:
                vector[5] = 1
    return vector
json_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive/litmus_vector_new_litmus.json"


def save_compact_json(data, filename):
    """
    保存 JSON 文件，但将纯数字的列表压缩到同一行显示。
    例如: [1, 2, 3] 不会分三行显示。
    """
    # 1. 先生成标准的、完全展开的 JSON 字符串
    text = json.dumps(data, indent=2)

    # 2. 定义正则：匹配 [ 数字/逗号/空格/换行 ] 的结构
    # Explanation:
    # \[           匹配左括号
    # \s* 匹配可能的空白
    # ([\d,\s\.\-]+?) 捕获组：匹配数字、逗号、空白、点、负号 (非贪婪匹配)
    # \s* 匹配可能的空白
    # \]           匹配右括号
    pattern = r'\[\s*([\d,\s\.\-]+?)\s*\]'

    # 3. 定义替换函数：把匹配到的内容里的“换行”和“多余空格”去掉
    def compress_list(match):
        # 获取 [ ] 里面的内容
        content = match.group(1)
        # 如果内容里包含 '{' 说明里面有对象，不是纯数字列表，跳过不压缩
        if '{' in content:
            return match.group(0)

        # 去掉换行和多余空格，并重新加上逗号后的空格
        # content.split(',') 会自动去掉换行符
        compact_content = ", ".join([x.strip() for x in content.split(',') if x.strip()])
        return f"[{compact_content}]"

    # 4. 执行替换
    text = re.sub(pattern, compress_list, text)

    # 5. 写入文件
    with open(filename, 'a+', encoding='utf-8') as f:
        f.write(text)

def extract_context_for_litmus(litmus_path, output_path):
    litmus_file = read_file(litmus_path)
    litmus = parse_litmus(litmus_file)
    litmus_name = litmus_path.split('/')[-1][:-7]
    config.init()
    config.set_var('reg_size', 64)
    rvwmo = RVWMO()
    rvwmo.run(litmus)
    for state in rvwmo.states:
        node_features = []
        edges = []
        node_map = {}
        node_id = 0
        ra = rvwmo.find_ra_by_state(state)
        exe = rvwmo.find_exe_by_state(state)
        for event in exe:
            node_map[(event.pid, event.idx)] = node_id
            node_features.append(get_vector_by_inst(event.inst.inst))
            node_id += 1
        rel_list = ["addr", "data", "ctrl", "ppo", "po", "rfe", "fr", "co", "fence"]
        for i,rel in enumerate(rel_list):
            rel_event_list = ra.find_all(rel)
            for e1, e2 in rel_event_list:
                if e1.inst is None or e2.inst is None:
                    continue
                edges.append([node_map[(e1.pid,e1.idx)], node_map[(e2.pid,e2.idx)], i])
        for (pid, idx) in node_map.items():
            print(pid, idx)
        for vec in node_features:
            print(vec)
        for edge in edges:
            print(edge)
        item = {
            "name": litmus_name,
            "node_features": node_features,
            "edges": edges,
            "state": str(state.state),
        }
        save_compact_json([item], output_path)
    # init_state = rvwmo.states[0]
    # ra = rvwmo.find_ra_by_state(init_state)
    # print(litmus)
    # # vector = [R-W,R-R,W-W,W-R,amo,lr/sc,aq,rl,addr,ctrl,data,fencecan,fencecannot]
    #
    # R = 0
    # W = 0
    # amo = 0
    # lr_sc = 0
    # aq = 0
    # rl = 0
    # addr = 1
    # ctrl = 1
    # data = 1
    # addr_list = ra.find_all("addr")
    # addr_idx_list = {}
    # for e1,e2 in addr_list:
    #     if e1.inst is None or e2.inst is None:
    #         continue
    #     addr_idx_list.setdefault(e1.pid, [])
    #     addr_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    # print("addr_idx_list", addr_idx_list)
    # data_list = ra.find_all("data")
    # data_idx_list = {}
    # for e1,e2 in data_list:
    #     if e1.inst is None or e2.inst is None:
    #         continue
    #     data_idx_list.setdefault(e1.pid, [])
    #     data_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    # print("data_idx_list", data_idx_list)
    # ctrl_list = ra.find_all("ctrl")
    # ctrl_idx_list = {}
    # for e1,e2 in ctrl_list:
    #     if e1.inst is None or e2.inst is None:
    #         continue
    #     ctrl_idx_list.setdefault(e1.pid, [])
    #     ctrl_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    # print("ctrl_idx_list", ctrl_idx_list)
    # loc_list = ra.find_all("po_loc")
    # print("loc_list")
    # for e1,e2 in loc_list:
    #     print(e1, e2)
    # loc_idx_list = {}
    # for e1,e2 in loc_list:
    #     if e1.inst is None or e2.inst is None:
    #         continue
    #     loc_idx_list.setdefault(e1.pid, [])
    #     loc_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    # print("loc_idx_list", loc_idx_list)
    # fencecan = 0
    # fencecannot = 0
    # fencei = 0
    # # --------
    # item = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # item[6] = addr
    # item[7] = data
    # item[8] = ctrl
    # exe_str = ""
    #
    # for thread in litmus.progs:
    #     j = None
    #     j_str = ""
    #     j_tmp = None
    #     Fflag = False
    #
    #     for i, inst in enumerate(thread.insts):
    #         if j is not None:
    #             exe_str += j_str
    #             j_str = ""
    #         if isinstance(inst, StoreInst):
    #             j_str = "W"
    #             j_tmp = j
    #             j = i
    #         elif isinstance(inst, LoadInst):
    #             j_str = "R"
    #             j_tmp = j
    #             j = i
    #         elif isinstance(inst, FenceInst):
    #             exe_str += f" Fence.{inst.pre}.{inst.suc}"
    #             Fflag = True
    #             continue
    #         elif isinstance(inst, FenceTsoInst):
    #             exe_str += "Ftso"
    #             Fflag = True
    #             continue
    #         elif isinstance(inst, FenceIInst):
    #             exe_str += "Fi"
    #             Fflag = True
    #             continue
    #         elif isinstance(inst, AmoInst):
    #             if inst.type == IType.Amo:
    #                 if inst.flag == MoFlag.Strong:
    #                     j_str = "AmoSC"
    #                 elif inst.flag == MoFlag.Acquire:
    #                     j_str = "AmoAQ"
    #                 elif inst.flag == MoFlag.Release:
    #                     j_str = "AmoRL"
    #                 elif inst.flag == MoFlag.Relax:
    #                     j_str = "Amo"
    #             elif inst.type == IType.Lr or inst.type == IType.Sc:
    #                 if inst.type == IType.Lr:
    #                     j_str = "Lr"
    #                 elif inst.type == IType.Sc:
    #                     j_str = "Sc"
    #                 if inst.flag == MoFlag.Strong:
    #                     j_str = "SC"
    #                 elif inst.flag == MoFlag.Acquire:
    #                     j_str = "AQ"
    #                 elif inst.flag == MoFlag.Release:
    #                     j_str = "RL"
    #                 elif inst.flag == MoFlag.Relax:
    #                     j_str = ""
    #             j_tmp = j
    #             j = i
    #         else:
    #             continue
    #         exe_str += ' '
    #
    #         if j_tmp == None:
    #             continue
    #         if Fflag:
    #             Fflag = False
    #             continue
    #
    #         if (j_tmp, i) in addr_idx_list.get(inst.pid, []):
    #             exe_str += "Addr"
    #         elif (j_tmp, i) in data_idx_list.get(inst.pid, []):
    #             exe_str += "Data"
    #         elif (j_tmp, i) in ctrl_idx_list.get(inst.pid, []):
    #             exe_str += "Ctrl"
    #
    #         if (j_tmp, i) in loc_idx_list.get(inst.pid, []):
    #             exe_str += "PoLoc"
    #         else:
    #             exe_str += "Po"
    #         exe_str += ' '
    #     if j is not None:
    #         exe_str += j_str
    #         j_str = ""
    #
    #
    #     exe_str += ' <SEP> '
    return ""

if __name__ == "__main__":
    # litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive'
    litmus_dir_path = '/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output'
    litmus_vector_log = os.path.join(litmus_dir_path, 'litmus_vector11.log')
    litmus_paths = get_files(litmus_dir_path, ".litmus")

    litmus_paths = litmus_paths[:]
    litmus_dict = {}
    # with open(litmus_vector_log, 'w') as f:
    #     pass
    for litmus_path in litmus_paths:
        print(litmus_path)
        litmus_name = litmus_path.split('/')[-1][:-7]
        if litmus_name in ["ISA03+SB01","ISA03+SB02","SWAP-LR-SC","ISA03","ISA03+SIMPLE","ISA03+SIMPLE+BIS"]:
            continue
        extract_context_for_litmus(litmus_path,litmus_vector_log)

