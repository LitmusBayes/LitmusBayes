import math
import re

from numpy.testing.print_coercion_tables import print_new_cast_table

from src.litmusbayes import config
from src.litmusbayes.analysis import RVWMO
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.prog import LoadInst, StoreInst, AmoInst, IType
from src.litmusbayes.utils.file_util import search_file, read_file

print_flag = True
mm = RVWMO()
config.init()
config.set_var('reg_size', 64)
def perpLE(litmus_test, asm_list = [],  mode = 'e'): # exhaustive or heuristic

    trans_dict = {}
    state_regs = []
    litmus_test = parse_litmus(litmus_test)
    out_regs_with_val = litmus_test.out_regs_with_val
    out_regs_with_val_map = {}
    for pid, reg_name, val in out_regs_with_val:
        out_regs_with_val_map[(pid, reg_name)] = val
        state_regs.append((pid,reg_name))
    if print_flag:
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
    for prog in litmus_test.progs:
        for inst in prog.insts:
            if print_flag:
                print(inst)
            if isinstance(inst, LoadInst) or (isinstance(inst, AmoInst) and inst.type != IType.Sc):
                if print_flag:
                    print('load', inst)
                    print('load', inst.idx, inst.pid)
                load_list.append((inst.pid, inst.idx))
                load_map[(inst.pid, inst.idx)] = load_counter
                load_reg_map[(inst.pid, str(inst.rd))] = load_counter
                load_counter += 1
            if isinstance(inst, StoreInst) or (isinstance(inst, AmoInst) and inst.type != IType.Lr):
                if print_flag:
                    print('store', inst)
                    print('store', inst.idx, inst.pid)
                store_list.append((inst.pid, inst.idx))
                store_map[(inst.pid, inst.idx)] = store_counter
                store_counter += 1

    # get lcm(least common multiple)
    litmus_lcm = 1
    print(mm.states)
    for states in mm.states:
        for reg in states.state:
            val = int(states.state[reg].as_long())
            if val == 0 :
                continue
            litmus_lcm = math.lcm(val, litmus_lcm)
    print('litmus_lcm', litmus_lcm)
    print(litmus_test.location_regs)
    for states in mm.states:
        final_state_flag = True
        for reg in states.state:
            val = int(states.state[reg].as_long())
            reg_pid = int(reg.split(':')[0].strip())
            reg_id = reg.split(':')[1].strip()
            if (reg_pid, reg_id, val) not in out_regs_with_val:
                final_state_flag = False
                break
        if not final_state_flag:
            continue
        print(states)
    #
        state_result_list = []
        load_to_store_map = {}
        ra = mm.find_ra_by_state(states)
        rf_list = ra.find_all('rf')
        fr_list = ra.find_all('fr')
        co_list = ra.find_all('co')

        index_map = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        index_map = [f'perple_{index}' for index in index_map]
        conds = []
        change_store_regs = []
        # toposort
        in_edge_nums = {}
        for num in range(thread_num):
            in_edge_nums[num] = 0

        new_rf_list = []
        new_fr_list = []
        for inst_1, inst_2 in rf_list:
            # print("inst_2 ", (inst_2.pid, str(inst_2.inst.get_def()).split('_')[0]))
            # print("val ", out_regs_with_val_map)
            # print("in ", (inst_2.pid, str(inst_2.inst.get_def()).split('_')[0]) not in out_regs_with_val_map)
            if (inst_2.pid, str(inst_2.inst.get_def()).split('_')[0]) not in out_regs_with_val_map:
                continue
            new_rf_list.append((inst_1, inst_2))

        for inst_1, inst_2 in fr_list:
            # print("inst_1 ", (inst_1.pid, str(inst_1.inst.get_def()).split('_')[0]))
            # print("val ", out_regs_with_val_map)
            # print("in ", (inst_1.pid, str(inst_1.inst.get_def()).split('_')[0]) not in out_regs_with_val_map)
            if (inst_1.pid, str(inst_1.inst.get_def()).split('_')[0]) not in out_regs_with_val_map:
                continue
            new_fr_list.append((inst_1, inst_2))

        rf_list = new_rf_list
        fr_list = new_fr_list

        for inst_1, inst_2 in rf_list:
            if inst_1.idx == -1 or inst_2.idx == -1:
                continue
            if inst_1.pid == inst_2.pid:
                continue
            print(f"{inst_1} -rf> {inst_2}")
            in_edge_nums[inst_1.pid] += 1


        # for fr
        for inst_1, inst_2 in fr_list:
            if inst_1.idx == -1 or inst_2.idx == -1:
                continue
            if inst_1.pid == inst_2.pid:
                continue
            store_sig = store_map[(inst_2.pid, inst_2.idx)]
            load_sig = load_map[(inst_1.pid, inst_1.idx)]
            load_tid = inst_1.pid
            store_tid = inst_2.pid
            in_edge_nums[inst_2.pid] += 1
            print(f"{inst_1} -fr> {inst_2}")

        # for rf

        change_thread_list = []
        init_thread_list = []
        for tid in in_edge_nums:
            if in_edge_nums[tid] == 0:
                change_thread_list.append(tid)
                init_thread_list.append(tid)
        if len(change_thread_list) == 0:
            change_thread_list.append(0)
            init_thread_list.append(0)

        if len(change_thread_list) > 1:
            with open("ex.txt", "a+") as f:
                f.write(litmus_test.name)
                f.write("\n")
        # for fr
        edge_nums = 0
        has_add_rf_list = []
        has_add_fr_list = []
        # assert len(change_thread_list) == 1, in_edge_nums

        ctx_list = []

        while (True):
            for inst_1, inst_2 in rf_list:
                if (inst_1, inst_2) in has_add_rf_list:
                    continue
                if inst_1.idx == -1 or inst_2.idx == -1:
                    edge_nums += 1
                    has_add_rf_list.append((inst_1, inst_2))
                    continue
                if inst_1.pid == inst_2.pid:
                    edge_nums += 1
                    has_add_rf_list.append((inst_1, inst_2))
                    continue
                store_sig = store_map[(inst_1.pid, inst_1.idx)]
                load_sig = load_map[(inst_2.pid, inst_2.idx)]
                load_tid = inst_2.pid
                store_tid = inst_1.pid
                load_reg_name = str(inst_2.inst.get_def()).split('_')[0].strip()
                store_reg = str(inst_1.inst.get_data())
                store_val = ra.exe_value_map[store_reg]
                # store_addr_reg = str(inst_1.inst.get_addr())

                store_reg = store_reg.split('_')[0].strip()
                # store_addr_reg = store_addr_reg.split('_')[0].strip()
                # change_store_regs.append((store_tid, store_reg, store_addr_reg))
                change_store_regs.append((store_tid, store_reg))

                print('rf', inst_1, inst_2, store_sig, load_sig)

                load_to_store_map.setdefault(load_sig, []).append(store_sig)  # rf: >
                co_flag = False
                for co_1, co_2 in co_list :
                    if co_2.inst == None or co_1.inst == None:
                        continue
                    if isinstance(co_1.inst.inst, AmoInst) or isinstance(co_2.inst.inst, AmoInst): # to fix
                        continue
                    if (co_1 == inst_1 and co_2.inst != None) or (co_2 == inst_1 and co_1.inst != None):
                        co_flag = True
                if litmus_lcm > 1:
                    # store_val = store_val % litmus_lcm
                    if f'ctx.out_{load_tid}_{load_reg_name}' not in ctx_list:
                        ctx_list.append(f'ctx.out_{load_tid}_{load_reg_name}')
                    conds.append(
                        f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}] % {litmus_lcm} == {store_val % litmus_lcm}')

                if mode == 'h':
                    if f'ctx.out_{load_tid}_{load_reg_name}' not in ctx_list:
                        ctx_list.append(f'ctx.out_{load_tid}_{load_reg_name}')
                    if store_tid not in change_thread_list and load_tid in change_thread_list:
                        index_map[
                            store_tid] = f'(ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}] - {store_val}) / {litmus_lcm}'
                        change_thread_list.append(store_tid)
                        edge_nums += 1
                        has_add_rf_list.append((inst_1, inst_2))
                    elif load_tid in change_thread_list:

                        if co_flag:
                            conds.append(
                                f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}]<={litmus_lcm}*({index_map[store_tid]})+{store_val}')
                            conds.append(f'{index_map[load_tid]}< _b->size_of_test')
                        else:
                            conds.append(
                                f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}]>={litmus_lcm}*({index_map[store_tid]})+{store_val}')
                            conds.append(f'{index_map[load_tid]}< _b->size_of_test')
                        edge_nums += 1
                        has_add_rf_list.append((inst_1, inst_2))

                elif mode == 'e':
                    if f'ctx.out_{load_tid}_{load_reg_name}' not in ctx_list:
                        ctx_list.append(f'ctx.out_{load_tid}_{load_reg_name}')
                    if co_flag:
                        conds.append(
                            f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}]<={litmus_lcm}*({index_map[store_tid]})+{store_val}')
                    else:
                        conds.append(
                            f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}]>={litmus_lcm}*({index_map[store_tid]})+{store_val}')
                    edge_nums += 1
                    has_add_rf_list.append((inst_1, inst_2))

            for inst_1, inst_2 in fr_list:
                if (inst_1, inst_2) in has_add_fr_list:
                    continue
                if inst_1.idx == -1 or inst_2.idx == -1:
                    edge_nums += 1
                    has_add_fr_list.append((inst_1, inst_2))
                    continue
                if inst_1.pid == inst_2.pid:
                    edge_nums += 1
                    has_add_fr_list.append((inst_1, inst_2))
                    continue
                store_sig = store_map[(inst_2.pid, inst_2.idx)]
                load_sig = load_map[(inst_1.pid, inst_1.idx)]
                load_tid = inst_1.pid
                store_tid = inst_2.pid
                load_reg_name = str(inst_1.inst.get_def()).split('_')[0].strip()
                store_reg = str(inst_2.inst.get_data())
                store_val = ra.exe_value_map[store_reg]
                # store_addr_reg = str(inst_2.inst.get_addr())

                store_reg = store_reg.split('_')[0].strip()
                # store_addr_reg = store_addr_reg.split('_')[0].strip()
                # change_store_regs.append((store_tid, store_reg, store_addr_reg))
                change_store_regs.append((store_tid, store_reg))


                load_to_store_map.setdefault(load_sig, []).append(-store_sig) # fr: <
                if f'ctx.out_{load_tid}_{load_reg_name}' not in ctx_list:
                    ctx_list.append(f'ctx.out_{load_tid}_{load_reg_name}')
                if litmus_lcm > 1:
                    # store_val = store_val % litmus_lcm
                    conds.append(f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}] % {litmus_lcm} == {store_val % litmus_lcm}')

                if mode == 'h':
                    if store_tid not in change_thread_list and load_tid in change_thread_list:
                        index_map[store_tid] = f'(ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}] - {store_val}) / {litmus_lcm} + 1'
                        change_thread_list.append(store_tid)
                        edge_nums += 1
                        has_add_fr_list.append((inst_1, inst_2))
                        conds.append(f'{index_map[load_tid]}< _b->size_of_test')
                    elif load_tid in change_thread_list:
                        conds.append(
                            f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}]<={litmus_lcm}*({index_map[store_tid]}-1)+{store_val}')
                        edge_nums += 1
                        has_add_fr_list.append((inst_1, inst_2))
                        conds.append(f'{index_map[load_tid]}< _b->size_of_test')
                elif mode == 'e':
                    conds.append(
                        f'ctx.out_{load_tid}_{load_reg_name}[{index_map[load_tid]}]<={litmus_lcm}*({index_map[store_tid]}-1)+{store_val}')
                    edge_nums += 1
                    has_add_fr_list.append((inst_1, inst_2))

                first_flag = True

            if edge_nums == len(rf_list) + len(fr_list):
                break


        for cond in conds:
            print(cond)

        # create condition
        for i, reg_sig in enumerate(states.state):
            reg_value = states.state[reg_sig]
            pid, reg_id = int(reg_sig.split(":")[0]), reg_sig.split(":")[1]
            load_sig = load_reg_map[(pid, reg_id)]
            load_to_store_list = load_to_store_map.get(load_sig, [])
            print(load_to_store_list)


        lines = []

        if mode == 'e':
            # init
            count_var = "counter_exhaust"
            for var in ctx_list:
                lines.append(f"REVERSE_CTX({var}, _b->size_of_test);")
            lines.append(f"int {count_var} = 0;")

            # loop
            indent = ""
            thread_map = index_map[:thread_num]
            for var in thread_map:
                lines.append(f"{indent}for (int {var} = 0; {var} <= _b->size_of_test - 1 ; {var}++){{")
                indent += "  "

            cond_str = ' && '.join(conds)
            lines.append(f"{indent}if ({cond_str}){{")
            lines.append(f"{indent}  {count_var} += 1;")
            lines.append(f"{indent}}}")

            # close the for-loops
            for _ in thread_map[::-1]:
                indent = indent[:-2]
                lines.append(f"{indent}}}")
            lines.append(f'printf("exhaust statistic: %d\\n", counter_exhaust);')
            trans_dict['exhaust'] = '\n'.join(lines)
            print('\n'.join(lines))
        elif mode == 'h':
            # init
            count_var = "counter_heuristic"
            for var in ctx_list:
                lines.append(f"REVERSE_CTX({var}, _b->size_of_test);")
            lines.append(f"int {count_var} = 0;")

            # loop
            indent = ""
            thread_map = init_thread_list
            for var in thread_map:
                lines.append(f"{indent}for (int {index_map[var]} = 0; {index_map[var]} <= _b->size_of_test - 1 ; {index_map[var]}++){{")
                indent += "  "

            cond_str = ' && '.join(conds)
            lines.append(f"{indent}if ({cond_str}){{")
            lines.append(f"{indent}  {count_var} += 1;")
            lines.append(f"{indent}}}")

            # close the for-loops
            for _ in thread_map[::-1]:
                indent = indent[:-2]
                lines.append(f"{indent}}}")
            lines.append(f'printf("heuristic statistic: %d\\n", {count_var});')
            trans_dict['exhaust'] = '\n'.join(lines)
            print('\n'.join(lines))


        # change asm
        print(change_store_regs)

        output = []
        counter = 0  # 用于递增 #_litmus_P0_?


        # eg: sw %[x5], 0(%[x6])
        # asm_code = """
        # #_litmus_P0_0
        # ori %[x5], x0, 1
        # #_litmus_P0_1
        # sw %[x5], 0(%[x6])
        # #_litmus_P0_2
        # fence w, w
        # #_litmus_P0_3
        # ori %[x7], x0, 1
        # #_litmus_P0_4
        # sw %[x7], 0(%[x8])
        # """
        # asm_code = """
        # #_litmus_P0_0\n\t
        # sw %[x5],0(%[x6])\n
        # #_litmus_P0_1\n\t
        # sw %[x5],0(%[x7])\n
        # #END _litmus_P0\n\t
        # :
        # :[x5] "r" (1), [x6] "r"( & _a->x[_i]), [x7] "r"( & _a->y[_i])
        # :"cc", "memory"
        # """

        for thread_index, asm_code in asm_list:
            print('===================asm code====================')
            print(asm_code)
            lines = asm_code.splitlines()


            def strip_quotes(l):
                l = l.strip()
                if l.startswith('"') and l.endswith('"'):
                    l = l[1:-1]
                if l.endswith(','):
                    l = l[:-1]
                return l.replace("\\t", "\t").replace("\\n", "")

            raw = [strip_quotes(l) for l in lines if l.strip()]
            print(raw)

            ori_re = re.compile(
                r'^\s*ori\s+'
                r'(?:%\[(?P<rd1>\w+)\]|(?P<rd2>\w+))\s*,\s*'
                r'(?:%\[(?P<rs1>\w+)\]|(?P<rs2>\w+))\s*,\s*'
                r'(?P<imm>-?\d+|0x[0-9A-Fa-f]+)(?P<rest>.*)$'
            )

            r_re = re.compile(r'\[\s*(\w+)\s*\]\s*"r"\s*\(\s*(-?\d+)\s*\)\s*,?')
            r_equal_re = re.compile(r'\[\s*(\w+)\s*\]\s*"=&[^"]*"\s*\(([^)]+)\)\s*,?')


            new_lines = []
            counter = 0
            ori_map = []
            ori_map_without_imm = []
            val_map = []
            r_list = []
            # create
            for l in raw:
                if l.startswith(f"#_litmus_P{thread_index}_"):
                    continue
                if l.startswith(f"#START"):
                    new_lines.append(f'"{l}\\n"')
                    continue
                if l.startswith(f"#END"):
                    new_lines.append(f'"{l.strip()}\\n\\t"')
                    continue
                if l.startswith(f"asm __") or \
                   l.startswith(f':"cc"') or \
                   l.startswith(f");"):
                    new_lines.append(f'{l}')
                    continue
                if l.strip() == "":
                    continue
                m = ori_re.match(l)

                if m:
                    rd = m.group('rd1') or m.group('rd2')
                    rs = m.group('rs1') or m.group('rs2')
                    imm = m.group('imm')
                    # if rs != 'x0' or (thread_index, rd) not in change_store_regs:
                    #     new_lines.append(f'"#_litmus_P{thread_index}_{counter}\\n\\t"')
                    #     counter += 1
                    #     new_lines.append(f'"{l}\\n"')
                    #     continue
                    if rs != 'x0' :
                        new_lines.append(f'"#_litmus_P{thread_index}_{counter}\\n\\t"')
                        counter += 1
                        new_lines.append(f'"{l}\\n"')
                        continue
                    new_lines.append(f'"#_litmus_P{thread_index}_{counter}\\n\\t"')
                    counter += 1
                    # new_lines.append(f'"addi %[{rd}], %[{rd}], {litmus_lcm}\\n"')
                    ori_map.append((thread_index, rd, imm))
                    ori_map_without_imm.append((thread_index, rd))
                else:
                    reg_val_name = "_val"

                    def repl(m):
                        reg, val = m.group(1), int(m.group(2))
                        if val != 0:
                            val_map.append((thread_index, reg, val))
                            return f'[{reg}] "r" ({reg}{reg_val_name}),'
                        else:
                            return f'[{reg}] "r" (0),'
                    l = r_re.sub(repl, l)
                    l = l.strip(',')
                    if '"r"' in l: # to fix, if not "r"
                        l += "".join(r_list)

                    def eq_repl(m):
                        reg, var = m.group(1), m.group(2)
                        if (thread_index, reg) in ori_map_without_imm:
                            r_list.append(f', [{reg}] "r" ({var})')
                            return f''
                        else:
                            return f'[{reg}] "=&r" ({var}),'
                    l = r_equal_re.sub(eq_repl, l)
                    l = l.strip(',')
                    pattern = re.compile(r'(&_a->x)\[\s*_i\s*\]')
                    l = pattern.sub(r'\1[_j]', l)
                    pattern = re.compile(r'(&_a->y)\[\s*_i\s*\]')
                    l = pattern.sub(r'\1[_j]', l)
                    pattern = re.compile(r'(&_a->z)\[\s*_i\s*\]')
                    l = pattern.sub(r'\1[_j]', l)
                    if l.startswith(':'):
                        new_lines.append(f'{l}')
                        continue
                    new_lines.append(f'"#_litmus_P{thread_index}_{counter}\\n\\t"')
                    counter += 1

                    new_lines.append(f'"{l}\\n"')
            def c_quote(l): return f'    "{l}\\n"'
            # c_style_asm = "\n".join(c_quote(l) for l in new_lines)

            c_style_asm = "\n".join(new_lines)
            print(c_style_asm)
            trans_dict[f'P{thread_index}']=c_style_asm
            trans_dict[f'P{thread_index}_init'] = ""
            trans_dict[f'P{thread_index}_var'] = []
            trans_dict[f'P{thread_index}_after_asm'] = ""
            for pid, reg_name, imm in ori_map:
                # trashed_str = f'int trashed_{reg_name} = {imm} + (_size_of_test - _j - _i)* {litmus_lcm};\n'
                trashed_str = f'int trashed_{reg_name} = {imm};\n'
                trashed_str2 = f'trashed_{reg_name} += {litmus_lcm};\n'
                print(trashed_str)
                trans_dict[f'P{thread_index}_init']+=f'{trashed_str}\n'
                trans_dict[f'P{thread_index}_var'].append(f'trashed_{reg_name}')
                trans_dict[f'P{thread_index}_after_asm']+=f'{trashed_str2}\n'

            for pid, reg_name, imm in val_map:
                val_str1 = f'int {reg_name}_val = {imm} ;\n'
                val_str2 = f'{reg_name}_val += {litmus_lcm};\n'
                trans_dict[f'P{thread_index}_init'] += f'{val_str1}\n'
                trans_dict[f'P{thread_index}_var'].append(f'trashed_{reg_name}')
                trans_dict[f'P{thread_index}_after_asm'] += f'{val_str2}\n'
                print(val_str1)
                print(val_str2)
    return trans_dict

if __name__ == "__main__":
    litmus_test_suite = ['MP']
    litmus_test_suite = [search_file(litmus, f'{config.TEST_DIR}/experiment/exp1_litmus', '.litmus') for litmus in litmus_test_suite]
    litmus_test_suite = [read_file(file_path) for file_path in litmus_test_suite]

    for litmus in litmus_test_suite:
        print(litmus)
        perpLE(litmus)