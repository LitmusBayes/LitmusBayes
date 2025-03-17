



import math
import os.path
import re
from itertools import product

from numpy.testing.print_coercion_tables import print_new_cast_table

from src.litmusbayes import config
from src.litmusbayes.analysis import RVWMO
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.litmus.litmus_changer import InjectList, InjectPoint
from src.litmusbayes.prog import LoadInst, StoreInst, FenceInst, IFmtInst
from src.litmusbayes.utils.file_util import search_file, read_file


mm = RVWMO()
config.init()
config.set_var('reg_size', 64)

delay_action_list = [
    [],
    ['fence'],
    ['load'],
    ['load', 'load'],
    ['load', 'load', 'load'],
    ['nop'],
    ['nop', 'nop'],
    ['nop', 'nop', 'nop'],
    ['nop', 'nop', 'nop', 'nop'],
    ['nop', 'nop', 'nop', 'nop', 'nop']
]


def generate_cartesian_lists(n, m):
    return [list(x) for x in product(range(m), repeat=n)]

def delayTest(litmus_test): # exhaustive or heuristic
    litmus_name = litmus_test.split('/')[-1].split('.')[0]
    trans_dict = {}
    litmus_test = read_file(litmus_test)
    litmus_test = parse_litmus(litmus_test)
    out_regs_with_val = litmus_test.out_regs_with_val
    out_regs_with_val_map = {}
    for pid, reg_name, val in out_regs_with_val:
        out_regs_with_val_map[(pid, reg_name)] = val

    print(out_regs_with_val)

    mm.run(litmus_test)
    load_list = []
    load_reg_map = {}
    load_map = {}
    load_counter = 0
    store_list = []
    store_map = {}
    store_counter = 0
    thread_num = len(litmus_test.progs)
    inst_list = []
    for prog in litmus_test.progs:
        for inst in prog.insts:
            print(inst)
            if isinstance(inst, LoadInst):
                print('load', inst)
                print('load', inst.idx, inst.pid)
                load_list.append((inst.pid, inst.idx))
                load_map[(inst.pid, inst.idx)] = load_counter
                load_reg_map[(inst.pid, str(inst.rd))] = load_counter
                load_counter += 1
                inst_list.append((inst.pid, inst.idx, inst.rs1))
            if isinstance(inst, StoreInst):
                print('store', inst)
                print('store', inst.idx, inst.pid)
                store_list.append((inst.pid, inst.idx))
                store_map[(inst.pid, inst.idx)] = store_counter
                inst_list.append((inst.pid, inst.idx, inst.rs1))
                store_counter += 1
    delayList = generate_cartesian_lists(4,10)
    num = 0
    for dList in delayList:
        delay_list = InjectList()
        for i, action in enumerate(dList):
            action_list = delay_action_list[action]
            pid, idx, rs1 = inst_list[i]
            idx-=1
            for item in action_list:
                if item == 'fence':
                    delay_list.add_inject_point(InjectPoint(pid, idx, FenceInst('fence')))
                if item == 'load':
                    delay_list.add_inject_point(InjectPoint(pid, idx, LoadInst('lw', 'x0', str(rs1), 0)))
                if item == 'nop':
                    delay_list.add_inject_point(InjectPoint(pid, idx, IFmtInst('addi','x0','x0',1)))
        num += 1
        print(dList)
        dList = [str(d) for d in dList]
        litmus_path = os.path.join(litmus_dir, f'{litmus_name}_{'_'.join(dList)}.litmus')
        litmus_test.mutate_new_litmus(delay_list, litmus_path)
        print(litmus_path)
litmus_dir = '/home/whq/Desktop/code_list/perple_test/delay_litmus'


if __name__ == "__main__":
    litmus_test_suite = ['MP','2+2W','SB','LB','S','R']
    litmus_test_suite = [search_file(litmus, f'{config.TEST_DIR}/experiment/exp1_litmus', '.litmus') for litmus in litmus_test_suite]
    # litmus_test_suite = [read_file(file_path) for file_path in litmus_test_suite]

    for litmus in litmus_test_suite:
        print(litmus)
        delayTest(litmus)