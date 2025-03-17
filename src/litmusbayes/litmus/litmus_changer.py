from enum import Enum

import networkx as nx
from matplotlib.pyplot import xcorr

from src.litmusbayes import config
from src.litmusbayes.analysis.model import RVWMO
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.litmus.litmus import AddrCond
from src.litmusbayes.prog import Inst, MemoryAccessInst
from src.litmusbayes.synth.diy7_generator import Cycle
from src.litmusbayes.synth.memory_relation import *
from src.litmusbayes.synth.ppo_def import SinglePPO
from src.litmusbayes.utils.file_util import search_file, read_file

print_flag = True
class InjectType(Enum):
    add = 0
    remove = 1
    change = 2

class InjectPoint:
    def __init__(self, pid, idx, inst:Inst, mode=InjectType.add):
        self.pid = pid
        self.idx = idx
        self.inst = inst
        self.mode = mode
        self.final = None

    def __repr__(self):
        return f"{self.pid},{self.idx},{self.inst.get_raw_str()}"

    def __eq__(self, other):
        if not isinstance(other, InjectPoint):
            return False
        if self.mode != other.mode:
            return False
        return self.pid == other.pid and self.idx == other.idx and self.inst.get_raw_str() == other.inst.get_raw_str()

    def set_final(self, final:str):
        self.final = final

class InjectList:
    def __init__(self):
        self.inject_list = []

    def add_inject_point(self, inject_point:InjectPoint):
        self.inject_list.append(inject_point)

    def add_by_list(self,inject_list):
        self.inject_list.extend(inject_list.inject_list)

    def __repr__(self):
        return f"{self.inject_list}"

    def __eq__(self, other):
        if not isinstance(other, InjectList):
            return False

        if len(self.inject_list) != len(other.inject_list):
            return False

        inject_list_map = {}
        for inject_point in self.inject_list:
            inject_list_map[str(inject_point)] = 0

        for inject_point in other.inject_list:
            inject_list_map[str(inject_point)] = 1

        for inject_point in inject_list_map:
            if inject_list_map[str(inject_point)] == 0:
                return False
        return True




def get_max_register(litmus):
    regs = litmus.regs
    reg_max = 0
    for tid, reg in regs:
        reg_index = int(str(reg)[1:])
        if reg_index > reg_max:
            reg_max = reg_index
    return reg_max


def get_inject_list_by_thread(thread_ppo, thread_inst_list, thread_id, thread_max_reg_id):
    inject_list = InjectList()
    num = len([relation for relation in thread_ppo if not relation.relation_flag])
    mem_inst_list = [i for i,inst in enumerate(thread_inst_list) if isinstance(inst,MemoryAccessInst) or isinstance(inst,AmoInst)]
    ppo_unary_relation_list = [relation for relation in thread_ppo if not relation.relation_flag]
    if len(mem_inst_list) != len(ppo_unary_relation_list) and len(mem_inst_list) != len(ppo_unary_relation_list) + 1:
        assert False, 'ppo not match thread_inst list'
    ppo_index = 0
    print('thread ppo', thread_ppo)
    for i in range(num):
        relation = thread_ppo[ppo_index]
        pre_inst = thread_inst_list[mem_inst_list[i]]
        if isinstance(relation, Lr):
            mod_inst = AmoInst(relation.inst_name,str(pre_inst.rd),'x0', str(pre_inst.rs1))
            inject_list.add_inject_point(InjectPoint(thread_id, mem_inst_list[i],inst=mod_inst,mode=InjectType.change))
        elif isinstance(relation, Sc):
            # if i == 0 or isinstance(thread_ppo[ppo_index - 2], Lr):
            #     fix_inst = AmoInst('lr.w',str(pre_inst.rs2),'x0', str(pre_inst.rs1))
            #     # fix_fence_inst = FenceInst('fence',suc = 'w')
            #     position = -1
            #     if i > 0:
            #         position = mem_inst_list[i-1]
            #     # inject_point_fence = InjectPoint(thread_id, position, inst = fix_fence_inst, mode = InjectType.add)
            #     # inject_list.add_inject_point(inject_point_fence)
            #     inject_point_lr = InjectPoint(thread_id, position, inst = fix_inst, mode = InjectType.add)
            #     inject_list.add_inject_point(inject_point_lr)
            thread_max_reg_id += 1
            mod_inst = AmoInst(relation.inst_name, f'x{thread_max_reg_id}', str(pre_inst.rs2), str(pre_inst.rs1))
            inject_point = InjectPoint(thread_id, mem_inst_list[i],inst=mod_inst,mode=InjectType.change)
            inject_point.set_final(f'{thread_id}:x{thread_max_reg_id}==1')
            inject_list.add_inject_point(inject_point)
        ppo_index += 2
    if print_flag:
        print('inject_list',inject_list)
    return inject_list

def check_ppo_match_thread(thread_ppo, thread_inst_list):
    if print_flag:
        print(f'check ppo {thread_ppo} is match {thread_inst_list}')
    num = len([relation for relation in thread_ppo if not relation.relation_flag])
    mem_inst_list = [i for i,inst in enumerate(thread_inst_list) if isinstance(inst,MemoryAccessInst) or isinstance(inst,AmoInst)]
    if print_flag:
        print('thread mem list', mem_inst_list)
        print(f'thread ppo :{thread_ppo},num:{num}')
    if not (len(mem_inst_list) == num or len(mem_inst_list) == num + 1):
        return False
    ppo_index = 0


    # match reg by binary inst
    equal_regs_list = []
    zero_regs_list = []
    for j ,thread_inst in enumerate(thread_inst_list):
        print(thread_inst, thread_inst.name)
        if thread_inst.name == 'xor':
            if str(thread_inst.rs1) == str(thread_inst.rs2):
                zero_regs_list.append(str(thread_inst.rd))
        if thread_inst.name == 'add':
            if str(thread_inst.rs1) in zero_regs_list:
                equal_regs_list.append((str(thread_inst.rd),str(thread_inst.rs2)))
                equal_regs_list.append((str(thread_inst.rs2),str(thread_inst.rd)))
            if str(thread_inst.rs2)in zero_regs_list:
                equal_regs_list.append((str(thread_inst.rd), str(thread_inst.rs1)))
                equal_regs_list.append((str(thread_inst.rs1), str(thread_inst.rd)))
    print(equal_regs_list)
    while(True):
        add_list = []
        for pair1 in equal_regs_list:
            for pair2 in equal_regs_list:
                if pair1[0] == pair2[0]:
                    if (pair1[1], pair2[1]) not in equal_regs_list and (pair1[1], pair2[1]) not in add_list:
                        add_list.append((pair1[1], pair2[1]))
                        add_list.append((pair2[1], pair1[1]))
        if len(add_list) > 0:
            equal_regs_list.extend(add_list)
        else:
            break
    for i in range(num):
        relation = thread_ppo[ppo_index]
        pre_inst = thread_inst_list[mem_inst_list[i]]
        if print_flag:
            print(f'check relation: {relation}, pre_inst: {pre_inst}')
        relation_flag = relation.static_check(pre_inst, None, None)
        if print_flag:
            print(f'relation_flag: {relation_flag}')
        if not relation_flag:
            return False
        if i == num - 1:
            continue
        binary_relation = thread_ppo[ppo_index + 1]
        after_inst = thread_inst_list[mem_inst_list[i + 1]]
        midden_insts = thread_inst_list[mem_inst_list[i]+1 : mem_inst_list[i + 1]]
        if len(midden_insts) == 0:
            midden_insts =None
        if print_flag:
            print(f'binary_relation: {binary_relation}, pre_inst: {pre_inst}, midden_insts: {midden_insts}, after_inst: {after_inst}')
        binary_flag = binary_relation.static_check(pre_inst, after_inst, midden_insts, equal_regs_list = equal_regs_list)
        if print_flag:
            print(f'binary_flag: {binary_flag}')
        if not binary_flag:
            return False
        ppo_index += 2
    if print_flag:
        print(f'ppo {thread_ppo} match {thread_inst_list}')
    return True


def match_cycle(litmus, cycle):
    # config.init()
    # config.set_var('reg_size', 64)
    # rvwmo = RVWMO()
    # rvwmo.run(litmus)

    progs = litmus.progs
    thread_insts = []
    for prog in progs:
        thread_insts.append(prog.insts)
    if print_flag:
        print('thread insts')
        for insts in thread_insts:
            print(insts)
    
        print('cycle: ',cycle)
    thread_list, connect_list = cycle.get_thread_ppos()
    if print_flag:
        print('thread_list: ')
        for thread_ppo in thread_list:
            print(thread_ppo)
        print('connect list:')
        for connect_relation in connect_list:
            print(connect_relation)
    thread_pair_dict = {}



    for i, thread_ppo in enumerate(thread_list):
        for j, thread_inst in enumerate(thread_insts):
            if check_ppo_match_thread(thread_ppo, thread_inst):
                if i not in thread_pair_dict:
                    thread_pair_dict[i] = []
                thread_pair_dict[i].append(j)
    if print_flag:
        print('thread_pair_list', thread_pair_dict)

    connect_pair_list = [] # first thread can connect second thread

    # check match
    for i in range(len(thread_insts)):
        if i not in thread_pair_dict:
            return []  # diyone7 create new litmus test to Fix
            assert False, "don't match this cycle and litmus test file"

    # process init cond
    init_list = litmus.init
    init_vars_dict = {}  # eg:{x:[(0,x6),(1,x5)]}
    for init_item in init_list:
        if isinstance(init_item, AddrCond):
            init_vars_dict[(init_item.pid,init_item.reg)] = init_item.var
    if print_flag:
        print('init_conds', init_vars_dict)

    # match connect operator
    queue = [] #(index, thread_index_list) index show now match ppo thread i, thread_index_list show have match thread's index
    for index in thread_pair_dict[0]:
        queue.append((1,[index]))
    
    thread_num = len(connect_list)
    while(True):
        if len(queue) == 0:
            return [] # diyone7 create new litmus test to Fix
            assert False, "don't match this cycle and litmus test file"
        
        index, thread_index_list = queue.pop(0) 
        if len(thread_index_list) == len(thread_insts) and index == 1:
            if print_flag:
                print(f'match {thread_index_list}')
            return thread_index_list
        
        thread_pair_list = thread_pair_dict[index]
        if print_flag:
            print(f'ppo {index} match thread list{thread_pair_list}')
        for after_index in thread_pair_list:
            if index !=0 and after_index in thread_index_list:
                continue
            thread_inst_i = thread_insts[thread_index_list[-1]]
            thread_inst_j = thread_insts[after_index]
            connect_index = thread_num - 1 if index ==0 else index - 1 
            connect_relation = connect_list[connect_index]

            mem_inst_list_i = [i for i, inst in enumerate(thread_inst_i) if
                                 isinstance(inst, MemoryAccessInst) or isinstance(inst, AmoInst)]
            before_ppo_index = index - 1 if index != 0 else thread_num - 1
            thread_ppo_unary_num = len([relation for relation in thread_list[before_ppo_index] if not relation.relation_flag])
            if print_flag:
                print(thread_ppo_unary_num)
            first_inst = thread_inst_i[mem_inst_list_i[thread_ppo_unary_num - 1]]

            mem_inst_list_j = [j for j, inst in enumerate(thread_inst_j) if
                                isinstance(inst, MemoryAccessInst) or isinstance(inst, AmoInst)]
            second_inst= thread_inst_j[mem_inst_list_j[0]]
            if print_flag:
                print(first_inst, second_inst, connect_relation)
            first_var = init_vars_dict.get((first_inst.pid,first_inst.rs1),None)
            second_var = init_vars_dict.get((second_inst.pid,second_inst.rs1),None)
            same_address_flag = first_var == second_var
            if print_flag:
                print(f'same address_flag: {same_address_flag}')
            if connect_relation.static_check(first_inst, second_inst, None , None, same_address = same_address_flag):
                if index == 0:
                    queue.append((1,thread_index_list))
                    continue
                index += 1
                if index == thread_num:
                    index = 0
                
                new_index_list = copy.deepcopy(thread_index_list)
                new_index_list.append(after_index)
                queue.append((index, new_index_list))
                if print_flag:
                    print((index, new_index_list))


if __name__ == '__main__':
    name = 'MP'
    file = search_file(name, 'D:\python\slide-deheng\slide-deheng/tests/input/litmus', '.litmus')
    content = read_file(file)
    data = parse_litmus(content)
    get_max_register(data)
    cycle = Cycle(SinglePPO([W(),Po(),W()]),SinglePPO([Rfe(),R(),Po(),R(),Fre()]))
    LrScCycle = Cycle(SinglePPO([Sc(),Po(),W()]),SinglePPO([Rfe(),Lr(),Po(),R(),Fre()]))
    match_cycle_list = match_cycle(data, cycle)
    print(match_cycle_list)
    print(cycle.to_diy_format())
    print(cycle.to_diy_format())
