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
    addr_idx_list = [1]*len(litmus.progs)
    for e1,e2 in addr_list:
        addr_idx_list[e1.pid] += 1
    for idx in addr_idx_list:
        addr *= idx
    data_list = ra.find_all("data")
    data_idx_list = [1]*len(litmus.progs)
    for e1,e2 in data_list:
        data_idx_list[e1.pid] += 1
    for idx in data_idx_list:
        data *= idx
    ctrl_list = ra.find_all("ctrl")
    ctrl_idx_list = [1]*len(litmus.progs)
    for e1,e2 in ctrl_list:
        ctrl_idx_list[e1.pid] += 1
    for idx in ctrl_idx_list:
        ctrl *= idx
    fencecan = 0
    fencecannot = 0
    fencei = 0
    # --------
    item = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    item[6] = addr
    item[7] = data
    item[8] = ctrl
    for thread in litmus.progs:
        R = 0
        W = 0
        amo = 0
        lr_sc = 0
        aq = 0
        rl = 0
        fencecan = 0
        fencecannot = 0
        fencei = 0
        # print(thread)
        has_seen_W = 0
        has_seen_R = 0
        fence_list = []
        fence_tso_list = []
        R_W = 0
        W_R = 0
        for inst in thread.insts:
            # print(inst)
            if isinstance(inst, StoreInst):
                new_fence_list = []
                new_fence_tso_list = []
                W += 1
                R_W += has_seen_R
                for fence in fence_list:
                    if 'w' in fence.suc:
                        fencecan += 1
                    else:
                        new_fence_list.append(fence)
                for fence in fence_tso_list:
                    fencecan += 1
                fence_list = new_fence_list
                fence_tso_list = new_fence_tso_list
                has_seen_W += 1
            elif isinstance(inst, LoadInst):
                R += 1
                W_R += has_seen_W
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
            elif isinstance(inst, FenceTsoInst):
                if has_seen_R > 0:
                    fencecan += 1
                else:
                    fence_tso_list.append(inst)
            elif isinstance(inst, FenceIInst):
                fencei+=1
            elif isinstance(inst, AmoInst):
                if inst.type == IType.Amo:
                    amo += 1
                if inst.type == IType.Lr or inst.type == IType.Sc:
                    lr_sc += 1
                if inst.flag == MoFlag.Strong or inst.flag == MoFlag.Release:
                    rl += 1
                if inst.flag == MoFlag.Strong or inst.flag == MoFlag.Acquire:
                    aq += 1

        fencecannot += len(fence_list) + len(fence_tso_list)
        item[0] *= (R+1)
        item[1] *= (W+1)
        item[2] *= (amo+1)
        item[3] *= (lr_sc+1)
        item[4] *= (aq+1)
        item[5] *= (rl+1)
        item[9] *= (fencecan+1)
        item[10] *= (fencecannot+1)
        item[11] *= (fencei+1)
        item[12] *= (W_R)
        item[13] *= (R_W)
    vector = item
    print(vector)
    return vector


if __name__ == "__main__":
    litmus_dir_path = '/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive'
    litmus_vector_log = os.path.join(litmus_dir_path, 'litmus_vector.log')
    litmus_paths = get_files(litmus_dir_path)

    litmus_paths = litmus_paths[885:]
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