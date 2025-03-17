import os
from time import process_time_ns

from src.litmusbayes import config
from src.litmusbayes.analysis import RVWMO
from src.litmusbayes.bayes.util import get_files
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.prog import StoreInst, LoadInst, FenceInst, AmoInst, MoFlag, IType, FenceTsoInst, FenceIInst
from src.litmusbayes.utils.file_util import read_file


def extract_context_for_litmus(litmus_path):
    litmus_file = read_file(litmus_path)
    litmus = parse_litmus(litmus_file)
    config.init()
    config.set_var('reg_size', 64)
    rvwmo = RVWMO()
    rvwmo.run(litmus)
    init_state = rvwmo.states[0]
    ra = rvwmo.find_ra_by_state(init_state)
    print(litmus)
    # vector = [R-W,R-R,W-W,W-R,amo,lr/sc,aq,rl,addr,ctrl,data,fencecan,fencecannot]

    R = 0
    W = 0
    amo = 0
    lr_sc = 0
    aq = 0
    rl = 0
    addr = 1
    ctrl = 1
    data = 1
    addr_list = ra.find_all("addr")
    addr_idx_list = {}
    for e1,e2 in addr_list:
        if e1.inst is None or e2.inst is None:
            continue
        addr_idx_list.setdefault(e1.pid, [])
        addr_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    print("addr_idx_list", addr_idx_list)
    data_list = ra.find_all("data")
    data_idx_list = {}
    for e1,e2 in data_list:
        if e1.inst is None or e2.inst is None:
            continue
        data_idx_list.setdefault(e1.pid, [])
        data_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    print("data_idx_list", data_idx_list)
    ctrl_list = ra.find_all("ctrl")
    ctrl_idx_list = {}
    for e1,e2 in ctrl_list:
        if e1.inst is None or e2.inst is None:
            continue
        ctrl_idx_list.setdefault(e1.pid, [])
        ctrl_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    print("ctrl_idx_list", ctrl_idx_list)
    loc_list = ra.find_all("po_loc")
    print("loc_list")
    for e1,e2 in loc_list:
        print(e1, e2)
    loc_idx_list = {}
    for e1,e2 in loc_list:
        if e1.inst is None or e2.inst is None:
            continue
        loc_idx_list.setdefault(e1.pid, [])
        loc_idx_list[e1.pid].append((e1.inst.inst.idx, e2.inst.inst.idx))
    print("loc_idx_list", loc_idx_list)
    fencecan = 0
    fencecannot = 0
    fencei = 0
    # --------
    item = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    item[6] = addr
    item[7] = data
    item[8] = ctrl
    exe_str = ""

    for thread in litmus.progs:
        j = None
        j_str = ""
        j_tmp = None
        Fflag = False

        for i, inst in enumerate(thread.insts):
            if j is not None:
                exe_str += j_str
                j_str = ""
            if isinstance(inst, StoreInst):
                j_str = "W"
                j_tmp = j
                j = i
            elif isinstance(inst, LoadInst):
                j_str = "R"
                j_tmp = j
                j = i
            elif isinstance(inst, FenceInst):
                exe_str += f" Fence.{inst.pre}.{inst.suc}"
                Fflag = True
                continue
            elif isinstance(inst, FenceTsoInst):
                exe_str += " Ftso"
                Fflag = True
                continue
            elif isinstance(inst, FenceIInst):
                exe_str += " Fi"
                Fflag = True
                continue
            elif isinstance(inst, AmoInst):
                if inst.type == IType.Amo:
                    if inst.flag == MoFlag.Strong:
                        j_str = "AmoSC"
                    elif inst.flag == MoFlag.Acquire:
                        j_str = "AmoAQ"
                    elif inst.flag == MoFlag.Release:
                        j_str = "AmoRL"
                    elif inst.flag == MoFlag.Relax:
                        j_str = "Amo"
                elif inst.type == IType.Lr or inst.type == IType.Sc:
                    if inst.type == IType.Lr:
                        j_str = "Lr"
                    elif inst.type == IType.Sc:
                        j_str = "Sc"
                    if inst.flag == MoFlag.Strong:
                        j_str += "SC"
                    elif inst.flag == MoFlag.Acquire:
                        j_str += "AQ"
                    elif inst.flag == MoFlag.Release:
                        j_str += "RL"
                    elif inst.flag == MoFlag.Relax:
                        j_str += ""
                j_tmp = j
                j = i
            else:
                continue
            exe_str += ' '

            if j_tmp == None:
                continue
            if Fflag:
                Fflag = False
                continue

            if (j_tmp, i) in addr_idx_list.get(inst.pid, []):
                exe_str += "Addr"
            elif (j_tmp, i) in data_idx_list.get(inst.pid, []):
                exe_str += "Data"
            elif (j_tmp, i) in ctrl_idx_list.get(inst.pid, []):
                exe_str += "Ctrl"

            if (j_tmp, i) in loc_idx_list.get(inst.pid, []):
                exe_str += "PoLoc"
            else:
                exe_str += "Po"
            exe_str += ' '
        if j is not None:
            exe_str += j_str
            j_str = ""


        exe_str += ' <SEP> '
    return exe_str

if __name__ == "__main__":
    # litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive'
    litmus_dir_path = '/home/whq/Desktop/code_list/slide/src/slide/bayes/make_new_litmus/litmus_output'
    litmus_vector_log = os.path.join(litmus_dir_path, 'litmus_vector_new_n_gram.log')
    litmus_paths = get_files(litmus_dir_path)

    litmus_paths = litmus_paths[:]
    litmus_dict = {}
    # with open(litmus_vector_log, 'w') as f:
    #     pass
    for litmus_path in litmus_paths:
        print(litmus_path)
        litmus_name = litmus_path.split('/')[-1][:-7]
        if litmus_name in ["ISA03+SB01","ISA03+SB02","SWAP-LR-SC","ISA03","ISA03+SIMPLE","ISA03+SIMPLE+BIS"]:
            continue
        vec = extract_context_for_litmus(litmus_path)

        litmus_dict[litmus_name] = vec
        with open(litmus_vector_log, 'a+') as f:
            f.write(f"{litmus_name}:{vec}\n")