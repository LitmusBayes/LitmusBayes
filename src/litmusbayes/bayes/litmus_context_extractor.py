import os
from time import process_time_ns

from src.litmusbayes import config
from src.litmusbayes.analysis import RVWMO
from src.litmusbayes.bayes.util import get_files
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.prog import StoreInst, LoadInst, FenceInst, AmoInst, MoFlag, IType
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
    R_W = 0
    R_R = 0
    W_W = 0
    W_R = 0
    amo = 0
    lr_sc = 0
    aq = 0
    rl = 0
    addr = len(ra.find_all('addr'))
    ctrl = len(ra.find_all('ctrl'))
    data = len(ra.find_all('data'))
    fencecan = 0
    fencecannot = 0
    # --------
    for thread in litmus.progs:
        # print(thread)
        has_seen_W = 0
        has_seen_R = 0
        fence_list = []
        for inst in thread.insts:
            # print(inst)
            if isinstance(inst, StoreInst):
                new_fence_list = []
                W_W += has_seen_W
                R_W += has_seen_R
                for fence in fence_list:
                    if 'w' in fence.suc:
                        fencecan += 1
                    else:
                        new_fence_list.append(fence)
                fence_list = new_fence_list
                has_seen_W += 1
            elif isinstance(inst, LoadInst):
                W_R += has_seen_W
                R_R += has_seen_R
                new_fence_list = []
                for fence in fence_list:
                    if 'r' in fence.suc:
                        fencecan += 1
                    else:
                        new_fence_list.append(fence)
                fence_list = new_fence_list
                has_seen_R += 1
            elif isinstance(inst, FenceInst):
                pre = inst.pre
                if (has_seen_R > 0 and 'r' in pre) or (has_seen_W > 0 and 'w' in pre):
                    fence_list.append(inst)
                else:
                    fencecannot += 1
            elif isinstance(inst, AmoInst):
                if inst.type == IType.Amo:
                    amo += 1
                if inst.type == IType.Lr or inst.type == IType.Sc:
                    lr_sc += 1
                if inst.flag == MoFlag.Strong or inst.flag == MoFlag.Release:
                    rl += 1
                if inst.flag == MoFlag.Strong or inst.flag == MoFlag.Acquire:
                    aq += 1
        fencecannot += len(fence_list)

    vector = [R_W, R_R, W_W, W_R, amo, lr_sc, aq, rl, addr, ctrl, data, fencecan, fencecannot]
    print(vector)
    return vector


if __name__ == "__main__":
    litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive'
    litmus_vector_log = os.path.join(litmus_dir_path, 'litmus_vector.log')
    litmus_paths = get_files(litmus_dir_path)
    litmus_paths = litmus_paths[790:]
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