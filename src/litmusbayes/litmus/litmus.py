

"""Litmus Test"""
import copy
import itertools
import re
from copy import deepcopy
from typing import List

import toolz as tz
from antlr4 import *
from z3 import BitVec, Solver

import time

from src.litmusbayes import config
from src.litmusbayes.litmus.parser.LitmusLexer import LitmusLexer
from src.litmusbayes.litmus.parser.LitmusListener import LitmusListener
from src.litmusbayes.litmus.parser.LitmusParser import LitmusParser
from src.litmusbayes.prog import find_reg_by_name, AmoInst, RFmtInst, IFmtInst, BFmtInst, JFmtInst, StoreInst, \
    LoadInst, JalrInst, parse_program, Label, FenceInst
from src.litmusbayes.prog.types import EType, QType



class LitmusResult:
    def __init__(self, name, states=None, pos_cnt=-1, neg_cnt=-1, time_cost=-1, litmus_code = ""):
        self.name = name
        self.states: List[LitmusState] = states
        self.pos_cnt, self.neg_cnt = pos_cnt, neg_cnt
        self.time_cost = time_cost
        self.litmus_code = litmus_code

    def union(self, other):
        assert isinstance(other, LitmusResult), f'[ERROR] other is not an instance of LitmusChipResult'
        # self.states.extend(other.states)
        for state in other.states:
            if state not in self.states:
                self.states.append(state)
            else:
                index = self.states.index(state)
                self.states[index].num += state.num
        self.pos_cnt += other.pos_cnt
        self.neg_cnt += other.neg_cnt
        self.time_cost += other.time_cost

    def __eq__(self, other):
        if isinstance(other, LitmusResult):
            if other.name == self.name:
                return True
        return False

    def get_state_list_by_num(self):
        return sorted(self.states, key=lambda s: s.num)


class LitmusState:
    def __init__(self, state: dict = None, num = 0):
        self.num = num
        self.state = state if state else {}
        self.signature = None

    def __repr__(self):
        if self.signature:
            return self.signature

        reg_state = [(k, v) for k, v in list(self.state.items()) if ':' in k]
        reg_state = sorted(reg_state, key=lambda x: str(len(x[0])) + x[0])
        var_state = [(k, v) for k, v in list(self.state.items()) if ':' not in k]
        var_state = sorted(var_state, key=lambda x: x[0])

        self.signature = ''.join([f'{k}={v}; ' for k, v in reg_state + var_state])
        return self.signature

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == other

    def get_num_str(self):
        return f'{self.num}>{str(self)}'

opmap = {
    r'/\\': ' and ',
    r'\\/': ' or ',
    r'~': ' not ',
    r'=': '==',
    r'true': 'True',
    r'false': 'False'
}


def preprocess(text: str):
    """Split .litmus to three parts: init, prog, cond
    init and cond are handled by Litmus parser
    prog is handled by Program parser (borrow from mappo)

    Example:
    RISCV SB
    {                           #init
        0:x5=1; 0:x6=x; 0:x8=y;
        1:x5=1; 1:x6=y; 1:x8=x;
    }
    P0          | P1          ; #prog
    sw x5,0(x6) | sw x5,0(x6) ;
    lw x7,0(x8) | lw x7,0(x8) ;

    exists (0:x7=0 /\ 1:x7=0)   #cond

    """

    # remove comments
    for comment in re.findall(r'\(\*.*?\*\)', text):
        text = text.replace(comment, '')

    init_start, init_end = text.index('{') - 1, text.index('}') + 1
    quantifiers = ['~exists', '~forall', 'exists', 'forall']
    cond_start = -1
    for q in quantifiers:
        if q in text:
            cond_start = text.find(q)
            break
    assert cond_start > 0, f'cannot find quantifiers in litmus test.'

    header = text[:init_start] + "\n"
    init = text[init_start:init_end]
    prog = text[init_end:cond_start]
    lines = list(prog.split('\n'))

    lines = [line for line in lines if line != '']
    lines = [line.split('|') for line in lines]
    n_thread = len(lines[0])
    progs = [''] * n_thread
    cond = ''
    for line in lines[1:]:
        if line[0].startswith('locations') or line[0].startswith('filter'):
            cond += line[0] + '\n'
        if len(line) == 1 and n_thread != 1:  # empty line: LB+addr+addrpx-poxp+VAR2
            continue
        for n in range(n_thread):
            if not (line[n].strip() == "" or line[n].strip() == ";"):  # discard pure '' or ';'
                progs[n] += line[n] + '\n'

    cond += text[cond_start:]
    return init, cond, progs, header


class Cond:
    pass


class RegCond(Cond):
    def __init__(self, pid, reg, val):
        """
        :param pid:
        :param reg:
        :param val: may be int or variable
        """
        super().__init__()
        self.pid = pid
        self.reg = find_reg_by_name(reg)
        self.val = val

    def __str__(self):
        return f"{self.pid}:{self.reg.name}={self.val}"

    def __repr__(self):
        return f"{self.pid}:{self.reg.name}={self.val}"


class AddrCond(Cond):
    def __init__(self, pid, reg, var):
        super().__init__()
        self.pid = pid
        self.reg = find_reg_by_name(reg)
        self.var = var

    def __str__(self):
        return f"{self.pid}:{self.reg.name}={self.var}"

    def __repr__(self):
        return f"{self.pid}:{self.reg.name}={self.var}"


class VarCond(Cond):
    def __init__(self, var, val):
        super().__init__()
        self.var = var
        self.val = val

    def __str__(self):
        return f"{self.var}={self.val}"

    def __repr__(self):
        return f"{self.var}={self.val}"


class Litmus:
    def __init__(self):
        self.init, self.final, self.progs = [], '', []
        self.filter = ''
        self.final_raw = ''
        self._vars, self._regs, self._tgts = None, None, None

        self._input_regs = []
        self._addr_regs = []

        self._all_out_regs = []  # also include vars
        self._out_regs = []  # remove duplicates
        self._out_regs_with_val = []  # with val

        self._all_regs = []
        self._trashed_regs = []
        self._location_regs, self._location_vars = [], []
        self._filter_regs, self._filter_vars = [], []

        self.quantifier = None
        self.name = ''
        self.header = ''
        self._init_events = None
        self._init_conds = None

        self.loc_val = {}

        self.visited_signatures = set()

    @property
    def n_threads(self):
        return len(self.progs)

    @property
    def init_conds(self):
        if self._init_conds:
            return self._init_conds
        solver = Solver()
        init_regs = []
        for pid, reg, value in self.input_regs:
            reg_sym = BitVec(f'{reg}_p{pid}_0', config.REG_SIZE)
            init_regs.append((pid, reg))
            solver.add(reg_sym == value)

        for pid, reg, var in self.addr_regs:
            init_regs.append((pid, reg))
            solver.add(BitVec(f'{reg}_p{pid}_0', config.REG_SIZE) == BitVec(f'&{var}', config.REG_SIZE))

        resting_regs = [reg for reg in self.regs if reg not in init_regs]

        for pid, reg in resting_regs:
            # if reg is 'x0':
                # continue
            solver.add(BitVec(f'{reg}_p{pid}_0', config.REG_SIZE) == 0)

        for var in self.vars:
            value = self.get_init_var_value(var)
            solver.add(BitVec(var, config.REG_SIZE) == value)

        self._init_conds = solver.assertions()

        return self._init_conds

    def gen_parallel_paths(self):
        from src.litmusbayes.analysis import CFG
        from src.litmusbayes.analysis.pp import ParallelPath
        paths_in_progs = [CFG(prog.insts).find_all_paths() for prog in self.progs]
        pps = []
        for p in list(itertools.product(*paths_in_progs)):
            pp = ParallelPath(list(p), self.init_conds)
            if pp.se.verify():
                pps.append(pp)
        return pps


    @property
    def vars(self) -> List[str]:
        if self._vars is not None:
            return self._vars
        addr_conds = list(filter(lambda s: isinstance(s, AddrCond), self.init))
        self._vars = list(set([c.var for c in addr_conds]))
        return self._vars

    @property
    def input_regs(self) -> List[tuple[int, str, int]]:
        """
        :return: e.g.,  [(0, 'x5', 1), (0, 'x7', 2), (1, 'x5', 3), (1, 'x7', 4)]
        """
        if self._input_regs:
            return self._input_regs
        self._input_regs = [(cond.pid, cond.reg.name, cond.val) for cond in self.init
                            if isinstance(cond, RegCond)]
        return self._input_regs

    @property
    def addr_regs(self) -> List[tuple[int, str, str]]:
        """
        :return: e.g.,  [(0, 'x5', 'x'), (0, 'x7', 'y'), (1, 'x5', 'x'), (1, 'x7', 'y')]
        """
        if self._addr_regs:
            return self._addr_regs
        self._addr_regs = [(cond.pid, cond.reg.name, cond.var) for cond in self.init
                           if isinstance(cond, AddrCond)]
        return self._addr_regs

    def get_tgts(self) -> List[str]:
        if self._tgts is not None:
            return self._tgts
        import re
        self._tgts = []
        regs = list(set(re.findall(r'\d:x\d\d?', self.final)))
        regs = sorted(regs)
        final = self.final[:]
        for reg in regs:
            if reg in final:
                self._tgts.append(reg)
                final = final.replace(reg, ' ')
        for var in self.vars:
            if var in final:
                self._tgts.append(var)
                final = final.replace(var, ' ')

        return self._tgts

    @property
    def out_vars(self) -> List[str]:
        import re
        elements = re.split(r"[^a-zA-Z0-9]", self.final)
        return [var for var in self.vars if var in elements]

    @property
    def out_regs(self) -> List[tuple[int, str]]:
        """
        :return:  e.g., (1, 'x9'), (1, 'x8')
        """
        if self._out_regs:
            return self._out_regs
        for reg_or_var in self._all_out_regs:
            if isinstance(reg_or_var, RegCond):
                self._out_regs.append((reg_or_var.pid, reg_or_var.reg.name))
            # elif isinstance(reg_or_var, VarCond):
            #     self.out_regs.append(tuple(reg_or_var.var))
        self._out_regs = list(set(self._out_regs))
        return self._out_regs

    @property
    def out_regs_with_val(self) -> List[tuple[int, str, int]]:
        if self._out_regs_with_val:
            return self._out_regs_with_val
        for reg_or_var in self._all_out_regs:
            if isinstance(reg_or_var, RegCond):
                self._out_regs_with_val.append((reg_or_var.pid, reg_or_var.reg.name, reg_or_var.val))
            # elif isinstance(reg_or_var, VarCond):
            #     self.out_regs.append(tuple(reg_or_var.var))
        self._out_regs_with_val = list(set(self._out_regs_with_val))
        return self._out_regs_with_val



    def traverse_regs(self):
        """
        get all regs and trashed regs
        """
        all_regs = []
        dst_regs = []
        for pid, prog in enumerate(self.progs):
            regs = []
            for inst in prog.insts:
                # rd, rs2, rs1
                if isinstance(inst, AmoInst) or isinstance(inst, RFmtInst):
                    regs.extend([inst.rd, inst.rs2, inst.rs1])
                    dst_regs.append((pid, inst.rd.name))
                # rd, rs1
                elif isinstance(inst, IFmtInst) or isinstance(inst, LoadInst) or isinstance(inst, JalrInst):
                    regs.extend([inst.rd, inst.rs1])
                    dst_regs.append((pid, inst.rd.name))
                # rs1, rs2
                elif isinstance(inst, BFmtInst) or isinstance(inst, StoreInst):
                    regs.extend([inst.rs1, inst.rs2])
                # rd
                elif isinstance(inst, JFmtInst):
                    regs.extend([inst.rd])
                    dst_regs.append((pid, inst.rd.name))
            regs = set([reg.name for reg in regs])
            all_regs.extend((pid, reg) for reg in regs)
        for reg in self.out_regs:
            if reg not in all_regs:
                all_regs.append(reg)
        self._all_regs = all_regs
        # out_reg_ids = [reg[1] for reg in self.get_out_regs()]
        self._trashed_regs = [reg for reg in set(dst_regs) if reg not in self.out_regs]

    @property
    def trashed_regs(self):
        return self._trashed_regs

    @property
    def location_vars(self):
        return self._location_vars

    @property
    def location_regs(self):
        return self._location_regs

    @property
    def filter_vars(self):
        return self._filter_vars

    @property
    def filter_regs(self):
        return self._filter_regs

    @property
    def regs(self):
        return self._all_regs

    def get_init_var_addr(self, pid: int, x: str):
        """
        query addrCond (0:x6 = x)
        :param pid:
        :param x: variable, e.g., x, y
        :return: reg name, e.g., x6
        """
        for cond in self.init:
            if isinstance(cond, AddrCond):
                if cond.var == x and cond.pid == pid:
                    return cond.reg.name
        return None

    def get_init_var_value(self, x: str):
        """
        query varCond (x = 5)
        :param x: variable, e.g., x, y
        :return: reg name, e.g., x6
        """
        if x not in self.vars:
            return None

        for cond in self.init:
            if isinstance(cond, VarCond):
                if cond.var == x:
                    return cond.val
        return 0

    def get_init_reg_value(self, pid: int, name: str):
        """
        query regCond (0:x5 = 1)
        :param pid:
        :param name:
        :return:
        """
        if name == 'x0' or name == 'zero':
            return 0

        for cond in self.init:
            if isinstance(cond, RegCond):
                if cond.pid == pid and cond.reg.name == name:
                    return cond.val
        return None

    #TODO:filter and location
    def mutate_new_litmus(self, inject_list, litmus_file_path, final_list = None):
        from src.litmusbayes.litmus.litmus_changer import InjectList, InjectType
        init_str = "{" + f"{'; '.join([str(cond) for cond in self.init])}" + "; }\n"

        # inject insts


        progs = copy.deepcopy(self.progs)
        progs_map = []
        for prog in progs:
            prog_array = prog.get_all_array() #Arrange all instructions, including Label
            progs_map.append(prog_array)
        for inject_point in inject_list.inject_list:
            prog_array = progs_map[inject_point.pid]
            if inject_point.idx == -1:
                prog_array.insert(0,inject_point.inst)
            for i,inst in enumerate(prog_array):
                if isinstance(inst, str):
                    continue
                if inject_point.idx == -1:
                    continue
                if inst.idx == inject_point.idx:
                    print(inst.idx,inst)
                    if inject_point.mode == InjectType.add:
                        inject_point.inst.idx = -1
                        prog_array.insert(i + 1, inject_point.inst)
                    elif inject_point.mode == InjectType.change:
                        idx = inst.idx
                        prog_array[i] = inject_point.inst
                        prog_array[i].idx = idx
                    elif inject_point.mode == InjectType.remove:
                        prog_array.pop(i)
                    else:
                        assert False, 'inject_type is error'
                    break

        progs_str = []
        for p in progs_map:
            prog = []
            for i,inst in enumerate(p):
                if isinstance(inst, Label) or isinstance(inst, str):
                    prog.append(str(inst).strip())
                else:
                    prog.append(inst.get_raw_str())
            progs_str.append(prog)
        # get max len of insts for fixed-length print
        max_lens = []
        for p in progs_str:
            max_lens.append(max([len(i) for i in p]) + 1)

        insts_size = max([len(p) for p in progs_str])  # get max number of insts
        pretty_progs = ""
        for i in range(self.n_threads):
            thread_name = f"P{i}"
            pretty_progs += f"{thread_name:^{max_lens[i] - 2}} | "
        pretty_progs = pretty_progs[:-2] + " ;\n"
        for i in range(insts_size):
            cur_line = ""
            for j in range(self.n_threads):
                if i < len(progs_str[j]):
                    if j == self.n_threads - 1:  # last thread
                        cur_line += f"{str(progs_str[j][i]):<{max_lens[j] - 1}}" + " ;"
                    else:
                        cur_line += f"{str(progs_str[j][i]):<{max_lens[j]}}" + "| "
                else:  # absence of inst
                    if j == self.n_threads - 1:  # last thread
                        cur_line += max_lens[j] * " " + ';'
                    else:
                        cur_line += max_lens[j] * " " + '| '
            pretty_progs += cur_line + "\n"
        quantifier_str = 'exists' if self.quantifier == QType.Exists else 'forall'

        final = copy.deepcopy(self.final)
        final = final.replace('not ','~ ')
        # final = final.replace('(','').replace(')','')
        for inject_point in inject_list.inject_list:
            if inject_point.final:
                final += ' and '+ inject_point.final

        final_str = str(final).replace('and','/\\').replace('or','\\/').replace('==','=')
        final_cond_str = f"{quantifier_str} {final_str}"

        if self.location_regs or self.location_vars:
            locations = f'locations '
            locations += f'['
            if self.location_vars:
                for var in self.location_vars:
                    locations += f'{var}; '
            if self.location_regs:
                for reg in self.location_regs:
                    locations += f'{reg[0]}:{reg[1]}; '
            locations += f']\n'
        else:
            locations = ''
        # print('--------------------------locations')
        # print(locations)
        filter_cond = f"\nfilter {self.filter.replace('and', '/\\').replace('or', '\\/').replace('==', '=')}\n" if self.filter else ''

        # time_str = str(time.time()).replace('.','_')
        # file_name = f'{litmus_dir}/{self.name}_{time_str}.litmus'
        header_str = self.header.split('\n')[0]
        # print(litmus_file_path)
        with open(litmus_file_path,'w') as f:
            # print('mutate new litmus')
            # print(f"{header_str}\n {init_str} {pretty_progs}{locations}{filter_cond}\n{final_cond_str}\n")
            f.write(f"{header_str}\n {init_str} {pretty_progs}{locations}{filter_cond}\n{final_cond_str}\n")

        return litmus_file_path

    def __repr__(self):
        # Test-driven development: MP+fence.rw.rw+ctrl.litmus. We have to add a new inst class: Label of branches
        init_str = "{" + f"{'; '.join([str(cond) for cond in self.init])}" + "}\n"
        progs_str = [str(p).split('\n') for p in self.progs]

        # get max len of insts for fixed-length print
        max_lens = []
        for p in progs_str:
            max_lens.append(max([len(i) for i in p]) + 1)

        insts_size = max([len(p) for p in progs_str])  # get max number of insts
        pretty_progs = ""
        for i in range(self.n_threads):
            thread_name = f"P{i}"
            pretty_progs += f"{thread_name:^{max_lens[i] - 2}} | "
        pretty_progs = pretty_progs[:-2] + " ;\n"
        for i in range(insts_size):
            cur_line = ""
            for j in range(self.n_threads):
                if i < len(progs_str[j]):
                    if j == self.n_threads - 1:  # last thread
                        cur_line += f"{str(progs_str[j][i]):<{max_lens[j] - 1}}" + " ;"
                    else:
                        cur_line += f"{str(progs_str[j][i]):<{max_lens[j]}}" + "| "
                else:  # absence of inst
                    if j == self.n_threads - 1:  # last thread
                        cur_line += max_lens[j] * " " + ';'
                    else:
                        cur_line += max_lens[j] * " " + '| '
            pretty_progs += cur_line + "\n"
        final_str = f"{self.quantifier} {self.final}"
        if self.location_regs or self.location_vars:
            locations = f'locations {self.location_vars + self.location_regs}\n'
        else:
            locations = ''

        filter_cond = f'\nfilter {self.filter}\n' if self.filter else ''
        return f"{init_str} {pretty_progs}{locations}{filter_cond}\n{final_str}\n"

    def update_final_str(self, new_final_str):
        # Test-driven development: MP+fence.rw.rw+ctrl.litmus. We have to add a new inst class: Label of branches
        init_str = "{" + f"{'; '.join([str(cond) for cond in self.init])}" + "}\n"
        progs_str = [str(p).split('\n') for p in self.progs]

        # get max len of insts for fixed-length print
        max_lens = []
        for p in progs_str:
            max_lens.append(max([len(i) for i in p]) + 1)

        insts_size = max([len(p) for p in progs_str])  # get max number of insts
        pretty_progs = ""
        for i in range(self.n_threads):
            thread_name = f"P{i}"
            pretty_progs += f"{thread_name:^{max_lens[i] - 2}} | "
        pretty_progs = pretty_progs[:-2] + " ;\n"
        for i in range(insts_size):
            cur_line = ""
            for j in range(self.n_threads):
                if i < len(progs_str[j]):
                    if j == self.n_threads - 1:  # last thread
                        cur_line += f"{str(progs_str[j][i]):<{max_lens[j] - 1}}" + " ;"
                    else:
                        cur_line += f"{str(progs_str[j][i]):<{max_lens[j]}}" + "| "
                else:  # absence of inst
                    if j == self.n_threads - 1:  # last thread
                        cur_line += max_lens[j] * " " + ';'
                    else:
                        cur_line += max_lens[j] * " " + '| '
            pretty_progs += cur_line + "\n"
        final_str = f"{self.quantifier} {self.final}"
        if self.location_regs or self.location_vars:
            locations = f'locations {self.location_vars + self.location_regs}\n'
        else:
            locations = ''

        filter_cond = f'\nfilter {self.filter}\n' if self.filter else ''
        return f"{init_str} {pretty_progs}{locations}{filter_cond}\n{new_final_str}\n"

    def eval_once(self, vals):
        """Evaluate one run"""
        tgts = self.get_tgts()
        assert len(tgts) == len(vals)
        final = self.final[:]
        for i, val in enumerate(vals):
            # to prevent from replacing 'a' in 'and', we use an extra character '='
            final = final.replace(f'{tgts[i]}=', str(val) + '=')
        return eval(final)

    def eval_all(self, vals_list):
        """Evaluate all runs"""
        return [self.eval_once(vals) for vals in vals_list]

    def eval_final(self, vals_list):
        """Evaluate all runs plus quantifier"""
        results = self.eval_all(vals_list)
        from operator import or_, and_
        if self.quantifier == QType.Forall:
            return tz.reduce(and_, results)
        elif self.quantifier == QType.Exists:
            return tz.reduce(or_, results)
        else:
            raise NotImplementedError

    def eval_filter(self, state):
        reg_state = [(f'{k[0]}:{k[1]}', str(v)) for k, v in state.items()]
        reg_state = sorted(reg_state, key=lambda x: str(len(x[0])), reverse=True)
        filter_cond = self.filter[:]
        for k, v in reg_state:
            filter_cond = filter_cond.replace(k, v)
        return eval(filter_cond)

    def set_header(self, header):
        self.name = re.findall(re.compile('RISCV (.*?)\n'), header)[0]
        self.header = header.strip()

    @property
    def init_events(self):
        if not self._init_events:
            from src.litmusbayes.analysis import Event
            self._init_events = []
            for var in self.vars:
                value = self.get_init_var_value(var)
                addr = BitVec(f'&{var}', config.REG_SIZE)
                self._init_events.append(Event(inst=None,
                                               etype=EType.Write,
                                               addr=addr, data=value, pid=-1))

        return self._init_events

    def get_cost_by_weight(self):
        cost_dict={
            AmoInst : 10,
            FenceInst : -1,
            StoreInst : 2,
            LoadInst : 2,
        }
        cost = 1
        for prog in self.progs:
            thread_cost = 1
            for inst in prog.insts:
                if type(inst) in cost_dict:
                    # print(inst, type(inst))
                    thread_cost += cost_dict[type(inst)]
            cost = cost * max(thread_cost, 1)
        return cost

parse_reg = lambda c: c.REG().getText()
parse_var = lambda c: c.VAR().getText()
parse_imm = lambda c: int(c.IMM().getText())
parse_pid = lambda c: int(c.PID().getText()[0:-1])


class LitmusParseListener(LitmusListener):

    def __init__(self):
        self.litmus = None

    def enterInit(self, ctx: LitmusParser.InitContext):

        reg_conds, var_conds, addr_conds = ctx.reg_cond(), ctx.var_cond(), ctx.addr_cond()

        if reg_conds is not None:
            for cond in reg_conds:
                assert cond.IMM() is not None, f'[ERROR] cond.IMM() is None. cond: {ctx.getText()}'
                pid, reg, imm = parse_pid(cond), parse_reg(cond), parse_imm(cond)
                self.litmus.init.append(RegCond(pid, reg, imm))

        if var_conds is not None:
            for cond in var_conds:
                if 'int' in cond.getText():
                    # print(f"[WARNING] unnecessary cond: {cond.getText()}, skipped.")
                    continue
                var, imm = parse_var(cond), parse_imm(cond)
                self.litmus.init.append(VarCond(var, imm))

        if addr_conds is not None:
            for cond in addr_conds:
                pid, reg, var = parse_pid(cond), parse_reg(cond), parse_var(cond)
                self.litmus.init.append(AddrCond(pid, reg, var))

    def enterFilter(self, ctx: LitmusParser.FilterContext):
        expr = ctx.cond_expr().getText()
        from src.litmusbayes.prog.reg import xregs_abi, xregs_numeric
        for i in range(31, -1, -1):
            expr = expr.replace(f':{xregs_abi[i]}', f':{xregs_numeric[i]}')

        for k, v in opmap.items():
            expr = expr.replace(k, v)

        self.litmus.filter = expr

    def enterFinal(self, ctx: LitmusParser.FinalContext):
        quantifier = None
        self.litmus.final_raw = ctx.getText()
        match ctx.QUANTIFIER().getText():
            case 'exists':
                quantifier = QType.Exists
            case 'forall':
                quantifier = QType.Forall

        expr = ctx.cond_expr().getText()

        for k, v in opmap.items():
            expr = expr.replace(k, v)

        if ctx.NOT() is not None:
            expr = f'not({expr})'
            quantifier_reverse = {
                QType.Exists: QType.Forall,
                QType.Forall: QType.Exists
            }
            quantifier = quantifier_reverse[quantifier]

        self.litmus.quantifier = quantifier
        self.litmus.final = expr

    def enterReg_cond(self, ctx: LitmusParser.Reg_condContext):
        parent = ctx.parentCtx
        while parent:
            if isinstance(parent, LitmusParser.FinalContext):
                pid, reg, imm = parse_pid(ctx), parse_reg(ctx), parse_imm(ctx)
                self.litmus._all_out_regs.append(RegCond(pid, reg, imm))
                return
            elif isinstance(parent, LitmusParser.FilterContext):
                pid, reg = parse_pid(ctx), parse_reg(ctx)
                self.litmus.filter_regs.append((pid, find_reg_by_name(reg)))
                return
            parent = parent.parentCtx

    def enterVar_cond(self, ctx: LitmusParser.Var_condContext):
        parent = ctx.parentCtx
        while parent:
            if isinstance(parent, LitmusParser.FinalContext):
                var = parse_var(ctx)
                imm = parse_imm(ctx)
                self.litmus._all_out_regs.append(VarCond(var, imm))
                return
            elif isinstance(parent, LitmusParser.FilterContext):
                var = parse_var(ctx)
                self.litmus.filter_vars.append(var)
                assert False, 'variables in filter condition is not supported for now.'
                return
            parent = parent.parentCtx

    def enterObserved_var(self, ctx: LitmusParser.Observed_varContext):
        if ctx.PID():
            pid, reg = parse_pid(ctx), parse_reg(ctx)
            self.litmus.location_regs.append((pid, find_reg_by_name(reg)))
        else:
            assert ctx.VAR()
            var = parse_var(ctx)
            self.litmus.location_vars.append(var)


def parse_litmus(text):
    litmus = Litmus()
    init, cond, progs, header = preprocess(text)
    litmus.set_header(header)
    litmus.progs = [parse_program(p) for p in progs]

    # set pid for each inst
    for pid, prog in enumerate(litmus.progs):
        for i in prog.insts:
            i.pid = pid

    lexer = LitmusLexer(InputStream(init + cond))
    stream = CommonTokenStream(lexer)
    parser = LitmusParser(stream)
    tree = parser.entry()
    listener = LitmusParseListener()
    listener.litmus = litmus  # assign litmus to listener
    walker = ParseTreeWalker()
    walker.walk(listener, tree)

    # actions
    litmus.traverse_regs()

    # FIX: only parse final condition
    # parser = LitmusParser(stream)
    # tree = parser.final()
    # listener.litmus = litmus
    # walker = ParseTreeWalker()
    # walker.walk(listener, tree)

    return listener.litmus
