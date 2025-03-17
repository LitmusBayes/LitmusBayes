

"""Memory Model"""
import copy
import itertools
import time
from typing import Generator, Tuple, Dict

import networkx as nx
import toolz as tz
from z3 import BitVec
from src.litmusbayes import config
from src.litmusbayes.analysis.pp import ParallelPath
from src.litmusbayes.analysis.rel import *
from src.litmusbayes.litmus import Litmus, LitmusState
from src.litmusbayes.log import *
from src.litmusbayes.utils.plot import AGraphNode, Cluster, plot_graph, AGraphEdge
# from slide.utils.plot import *
from src.litmusbayes.utils.progress import print_progress_bar

"""
Input: Litmus
Output: All Legal States

Algorithm:

1. Generate all parallel paths with initial conditions.
2. For each parallel path, generate all executions.
     a. Permutate events and put them in GMO (global memory order) as execution candidates.
     b. Find local ppos in each path.
     c. Abandon executions violating ppos.
     d. Return the resting legal executions.
3. For each execution, find RF (Read-From) relations and add the corresponding constraints:
   rf(w,r) => R.value == W.value.
4. Find final states by solving the constraints.
"""


def gen_executions_with_reads_permutation_reduction(events: list):
    """
    if we have 2 writes (o1 & o2) and 2 reads (x & x), there will 12 cases:
    x x o1 o2
    x o1 x o2
    x o1 o2 x
    o1 x x o2
    o1 x o2 x
    o1 o2 x x
    x x o2 o1
    x o2 x o1
    x o2 o1 x
    o2 x x o1
    o2 x o1 x
    o2 o1 x x
    instead of A(4,3) = 24 cases.
    """
    amo_reads = [e for e in events if e.type is EType.Read and e.inst and e.inst.name.startswith('amo')]
    writes = [events.index(e) for e in events if e.type is EType.Write]
    others = [events.index(e) for e in events if e not in amo_reads and e.type is not EType.Write]
    write_seqs = list(itertools.permutations(writes))
    n_writes = len(writes)
    other_seqs = [[[] for i in range(n_writes + 1)]]

    for o in others:
        other_seqs_new = []
        for ps in other_seqs:
            for pos in range(n_writes + 1):
                ps_new = copy.deepcopy(ps)
                ps_new[pos].append(o)
                other_seqs_new.append(ps_new)
        other_seqs = other_seqs_new[:]

    def merge(write_seq, other_seq):
        exe = []
        for i in range(n_writes):
            exe.extend(other_seq[i])
            exe.append(write_seq[i])
        exe.extend(other_seq[-1])
        return exe

    exes = [list(map(lambda x: events[x], exe))
            for exe in [merge(w, o) for w in write_seqs for o in other_seqs]]

    # attach amo read to the corresponding amo write
    for exe in exes:
        for amo_read in amo_reads:
            e = [e for e in exe if e.inst is amo_read.inst][0]
            exe.insert(exe.index(e), amo_read)
    # print('len(exes)', len(exes))
    return exes


def gen_exe_signature(rels: list) -> tuple:
    rels = [e1.signature + '->' + e2.signature for e1, e2 in rels]
    return tuple(sorted(rels))

print_flag=False
class MemoryModel:

    def __init__(self, ppo_g, ppo_l, plot_enabled: bool = False):
        """
        :param plot_enabled: if true, plot relation graph for each valid execution
        """
        no_ppo = lambda r, a, b: False
        self.ppo_g = ppo_g if ppo_g else no_ppo
        self.ppo_l = ppo_l if ppo_l else no_ppo
        self.states = []
        self._state_exe_map: Dict[LitmusState, Tuple[List[Event], GlobalRelationAnalyzer]] = {}
        self.plot_enabled = plot_enabled

    def find_ra_by_state(self, state) -> GlobalRelationAnalyzer:
        return self._state_exe_map[state][1]

    def find_exe_by_state(self, state) -> List[Event]:
        return self._state_exe_map[state][0]

    def gen_executions(self, pp, complete_flag=False, base_model=None) -> Generator[Tuple[List[Event], int, int], None, None]:
        """Generate legal executions from a ParallelPath
        Return: execution, index_of_execution, total_num_of_executions
        """
        #?
        ra = LocalRelationAnalyzer(ppo=self.ppo_l, path=pp.se.path, execution=pp.se.events, solver=pp.se.solver)
        events: List[Event] = pp.se.events
        # print('events',events)
        ppos = [(e1, e2) for e1 in events for e2 in events if self.ppo_l(ra, e1, e2)]

        # base ppo
        base_ppos = []
        if complete_flag:
            assert base_model is not None, 'need base model'
            base_ppos = [(e1, e2) for e1 in events for e2 in events if base_model.ppo_l(ra, e1, e2)]


        # print('ppos',ppos)
        # print('ppo_l',self.ppo_l)
        DEBUG(pp.paths, 'Path')
        DEBUG(pp.se.solver.assertions(), 'Constraints')

        DEBUG(events, 'Events')
        DEBUG(ppos, 'PPOs')

        if config.READS_PERMUTATION_REDUCTION:
            exes = gen_executions_with_reads_permutation_reduction(events)
        else:
            # total order
            exes = list(itertools.permutations(events))


        def violate_ppo(execution, complete_flag):
            """Check if an execution violate ppo."""
            gmo = lambda a, b: execution.index(a) < execution.index(b)
            for e1, e2 in ppos:
                if gmo(e2, e1):
                    if complete_flag and (e1, e2) in base_ppos:
                        return True
            return False

        def violate_amo(execution):
            gmo_idx = lambda x: execution.index(x)
            for e1 in [e for e in execution if e.type is ra.R(e) and e.is_amo()]:
                for e2 in [e for e in execution if e.type is ra.W(e) and e.is_amo()]:
                    # there is no event between the read and the write event from one amo inst.
                    if ra.amo(e1, e2) and gmo_idx(e2) - gmo_idx(e1) != 1:
                        return True
            return False

        DEBUG(exes, f'All Executions ({len(exes)})')

        # filter out paths which violate:
        # 1. ppo axiom (ppo is consistent with gmo)
        # 2. rmw axiom (amo is consistent with gmo)
        exes = [list(e) for e in exes if not violate_amo(e) and not violate_ppo(e, complete_flag)]

        DEBUG(exes, f'All Legal Executions ({len(exes)})')

        for i, exe in enumerate(exes):
            yield exe, i + 1, len(exes)

    def run(self, litmus, complete_flag = False, base_model = None) -> None:
        """
        Run Litmus test on MemoryModel.
        :param litmus: litmus test
        """
        print(litmus)
        output_states = []
        INFO(litmus.name, 'run litmus')
        start_time = time.time()
        total_num_exes = 0
        parallel_paths = litmus.gen_parallel_paths()
        # print('parllel_paths',parallel_paths)
        
        for pp_idx, pp in enumerate(parallel_paths):
            # print('pp_idx',pp_idx)
            # print('pp',pp)
            # Clear cached signatures. Actually, it is not necessary.
            # We do this for saving map space.
            litmus.visited_signatures.clear()
            exes = self.gen_executions(pp, complete_flag=complete_flag, base_model=base_model)
            try:
                while True:
                    exe, i, total = next(exes)
                    # print('exe',exe)
                    # print('i',i)
                    # print('total',total)
                    total_num_exes += 1

                    # print progress bar
                    if i % 100 == 0 or i == total:
                        print_progress_bar(i, total, prefix=f'Progress [{pp_idx + 1}/{len(parallel_paths)}]:',
                                           suffix=f'{i}/{total}', length=50, start_time=start_time)

                    # save the solver states
                    if print_flag:
                        print('model before',pp.se.solver.assertions())
                    pp.se.solver.push()
                    # run one execution and return the output state
                    if print_flag:
                        print('run one execution before')
                        print('pp',pp)
                        print('exe',exe)
                    state, ra = self.run_one_execution(litmus, pp, exe, complete_flag=complete_flag, base_model=base_model)
                    # if(state is not None):
                    #
                    #     print('exe')
                    #     for item in exe:
                    #         print(item)
                    #     print('state',state)
                    #     print('ra',ra)

                    if not state:
                        if print_flag:
                            print('model after',pp.se.solver.assertions())
                        del ra
                        pp.se.solver.pop()
                        continue

                    output_states.append(state)
                    self._state_exe_map[state] = (exe, ra)
                    if self.plot_enabled:
                        self.plot_execution(litmus, pp, exe, state)
                    if print_flag:
                        print('model after',pp.se.solver.assertions())
                    # recover the solver states
                    pp.se.solver.pop()
            except StopIteration:
                # do something after processing each parallel_path
                pass

        elapsed_time = "{: .4f}".format(time.time() - start_time)
        print('\n')  # newline for the progress bar

        # output states
        states = list(set(output_states))
        if len(states) == 0:
            # for fence.tso.litmus
            print(f'States 1')
            self.states.append(LitmusState())
            return

        print(f'States {len(states)}')
        states.sort(key=lambda s: str(s))
        for state in states:
            print(f'{state} => {output_states.count(state)}')

        # print statistics
        print(f'Time {litmus.name} {elapsed_time}s')
        print(f'#Executions: {total_num_exes}\n')

        # assert len(states) == len(self._state_exe_map.keys())
        self.states = states

        return

    def run_one_execution(self, litmus: Litmus, pp: ParallelPath, exe: List, complete_flag = False, base_model = None):
        # print('litmus',litmus)
        # print('pp',pp)
        # print('exe')
        # for item in exe:
        #     print(item)
        ra = GlobalRelationAnalyzer(ppo=self.ppo_g, path=pp.se.path, execution=litmus.init_events + exe,
                                    solver=pp.se.solver, litmus=litmus)
        base_ra = None
        if complete_flag:
            base_ra = GlobalRelationAnalyzer(ppo=base_model.ppo_g, path=pp.se.path, execution=litmus.init_events + exe,
                                    solver=pp.se.solver, litmus=litmus)
        solver = ra.solver
        # find rf for each read in gmo order and add 'r.value == w.value' to constraints
        reads = [e for e in ra.execution if ra.R(e)]
        rfs = []
        while len(reads) > 0:
            er = reads.pop(0)
            has_rf = False
            for ew in [e for e in ra.execution if ra.W(e)]:
                if ra.rf(ew, er):
                    solver.add(ew.value == er.value)
                    has_rf = True
                    rfs.append((ew, er))
                    ra.clear()
                    break
            # every read event must read from a write event
            if not has_rf:
                return None, None
        if print_flag:
            print('rfs', rfs)
        # while rf is determined, loc is also determined
        # we should avoid calling ra.clear() to accelerate checking

        # generate execution signature
        cos = ra.find_all('co')
        signature = gen_exe_signature(rfs + cos)
        if signature in litmus.visited_signatures:
            return None, None
        if print_flag:
            print('signature after')


        DEBUG(solver.assertions(), 'Constraints after rf')

        ppos = [
            (e1, e2)
            for e1 in ra.execution
            for e2 in ra.execution
            if ra.ppo(e1, e2)
        ]
        base_ppos = []
        if complete_flag:
            base_ppos = [
                (e1, e2)
                for e1 in ra.execution
                for e2 in ra.execution
                if base_ra.ppo(e1, e2)
            ]
        if print_flag:
            print('before check',)
        # global ppo axiom
        for e1, e2 in ppos:
            if ra.gmo(e2, e1):
                if print_flag:
                    print(e1,e2)
                if not complete_flag or (e1, e2) in base_ppos :
                    return None, None
        if print_flag:
            print('pass gmo')

        # atomic axiom
        for a, b in ra.find_all('rmw'):
            for c in ra.execution:
                if ra.fre(a, c) and ra.coe(c, b):
                    return None, None
        if print_flag:
            print('pass axiom')
        # sc succeed
        for b in [e for e in ra.execution if ra.W(e) and e.inst and e.inst.name.startswith('sc')]:
            lr_found = 0
            for a in [e for e in ra.execution if e.inst and e.inst.name.startswith('lr')]:
                if ra.po(a, b) and ra.loc(a, b):
                    lr_found = 1
                    # # The requirement for reordering is that the lr and sc instructions must be flipped together.
                    #
                    # # print('lr,sc')
                    # # for e in ra.execution:
                    # #     print(e)
                    # reorder_in_lr_sc_mid_list = [e for e in ra.execution if ra.execution.index(e)<ra.execution.index(b) and
                    #                                      ra.execution.index(e)>ra.execution.index(a) and
                    #                                      e.pid == a.pid
                    #           ]
                    # # print(reorder_in_lr_sc_mid_list)
                    # # print('lr id', a.idx)
                    # # for e in reorder_in_lr_sc_mid_list:
                    # #     print(e.idx)
                    # # print('sc id', b.idx)
                    # for e in reorder_in_lr_sc_mid_list:
                    #     if e.inst.idx > b.inst.idx:
                    #         return None, None

            if not lr_found:
                return None, None
        if print_flag:
            print('pass sc')
        # sc fail. it's useful for synthesis.
        # for a, b in ra.find_all(ra.rmw):
        #     if b.inst and b.type is EType.SC_Fail:
        #         fail_cause_found = False
        #         for c in [e for e in ra.execution if ra.W(e)]:
        #             if ra.gmo(a, c) and ra.gmo(c, b) and ra.loc(a, b):
        #                 fail_cause_found = True
        #                 break
        #         if not fail_cause_found:
        #             return None

        def acyclic(rels, acy_ra):
            all_rels = list(tz.concat([acy_ra.find_all(r) for r in rels]))
            all_rels_idx = list(set([(ra.execution.index(r[0]), ra.execution.index(r[1])) for r in all_rels]))
            if len(all_rels_idx) == 0:
                return True
            graph = nx.DiGraph()
            graph.add_edges_from(all_rels_idx)
            cycles = list(nx.simple_cycles(graph))
            return len(cycles) == 0

        # Main Model Axiom
        if not acyclic(['co', 'rfe', 'fr', 'ppo'], ra):
            if not complete_flag or not acyclic(['co', 'rfe', 'fr', 'ppo'], base_ra):
                
                return None, None
        if print_flag:
            print('pass main model axiom')
        # print('before sc per location')
        # SC Per Location
        # if not acyclic('co', 'rf', 'fr', 'po_loc'):
        #     return None, None
        if print_flag:
            print('pass sc per location')
            print('after valid test')

        result = solver.check()

        if not result == z3.sat:
            return None, None
        if print_flag:
            print('solver check success')
        sym_reg_map = {}

        def find_sym_for_reg(reg, model):
            prefix = f'{reg[1]}_p{reg[0]}_'
            candidates = [v for v in model.decls() if str(v).startswith(prefix)]
            assert len(candidates) > 0, f'Cannot find symbol for {reg[0]}:{reg[1]} in model'
            sym = max(candidates, key=lambda x: int(str(x).replace(prefix, '')))
            sym_reg_map[str(sym)] = reg
            return sym

        def sym_var_decl(name):
            return BitVec(name, config.REG_SIZE)
        if print_flag:
            print('model')
        # solve the constraints
        model = solver.model()
        DEBUG(model, 'Model')

        # filter state
        if litmus.filter:
            filter_reg = [find_sym_for_reg(reg, model) for reg in litmus.filter_regs]
            filter_reg_state = {sym_reg_map[str(sym)]: model[sym] for sym in filter_reg}
            if not litmus.eval_filter(filter_reg_state):
                return None, None

        # collect outcomes
        syms_out_reg = [find_sym_for_reg(reg, model) for reg in litmus.out_regs + litmus.location_regs]
        sym_value_map = {str(sym): model[sym] if sym in model.decls() else 0 for sym in syms_out_reg}
        ra.exe_value_map = {str(sym): model[sym].as_long() for sym in model.decls()}
        if print_flag:
            print('reg',litmus.out_regs+litmus.location_regs)
            print('syms_out_reg', syms_out_reg)
            print('sym_value_map', sym_value_map)
            print('model.decls', model.decls())
        print('reg',litmus.out_regs+litmus.location_regs)
        print('syms_out_reg', syms_out_reg)
        print('sym_value_map', sym_value_map)
        print('model.decls', model.decls())
        writes = [e for e in exe if ra.W(e)]
        writes.reverse()
        var_final_value = {}
        for var in litmus.out_vars + litmus.location_vars:
            DEBUG(var, f'search write event for {var}')
            for e in writes:
                DEBUG(e, 'check write event')
                if not ra.check(e.addr != sym_var_decl(f'&{var}')):
                    value = model.eval(e.value)
                    DEBUG(value, f'find the write event {e} for {var}, final value is')
                    var_final_value[var] = value
                    break
            # FIXME: why there is no write event for 'SWAP-LR-SC'?
            if var not in var_final_value:
                var_final_value[var] = 0
        if print_flag:
            print('var_final_value', var_final_value)
        DEBUG(var_final_value, 'out variables')

        # check unique
        # TODO: How to check unique?
        # for sym in model.decls():
        #     solver.add(var_decl(str(sym)) != model[sym])
        # DEBUG(solver.assertions(), 'Constraints before check unique')
        # if solver.check() != z3.unsat:
        #     print(f'Another solution is found: {solver.model()}')
        #     assert False, 'Solution is not unique!'

        # for sym, value in sym_value_map.items():
        #     solver.add(var_decl(sym) != value)
        # DEBUG(solver.assertions(), 'Constraints before check unique')
        # if solver.check() != z3.unsat:
        #     print(f'Another solution is found: {solver.model()}')
        #     assert False, 'Solution is not unique!'

        # collect outcomes
        def out_reg_format(reg):
            return f'{reg[0]}:{reg[1]}'

        def out_var_format(var):
            return f'[{var}]'

        state = {}
        for k, v in sym_value_map.items():
            state[out_reg_format(sym_reg_map[k])] = v
        for k, v in var_final_value.items():
            state[out_var_format(k)] = v
        if print_flag:
            print('sym_value_map', sym_value_map)
            print('var_final_value', var_final_value)

        # if the execution is valid, save its signature & location value
        litmus.visited_signatures.add(signature)
        for k, v in ra.loc_val.items():
            litmus.loc_val[k] = v
        if print_flag:
            print('loc_val', litmus.loc_val)
        # print(ra.execution)
        # print(LitmusState(state))
        return LitmusState(state), ra

    def plot_execution(self, litmus, pp, exe, state):
        """
        Plot the relation graph for one execution.
        :param litmus: litmus test.
        :param pp: parallel paths.
        :param exe: execution.
        :param state: output state for the execution.
        """

        # construct the filename for the graph.
        import datetime
        import random
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f") + str(random.randint(0, 100)) + '_state_' + str(state)
        filename = f'./output/{litmus.name}_{time_str}'

        # collect nodes
        exe = litmus.init_events + exe
        nodes = {e: AGraphNode(str(e)) for e in exe}

        edges = []

        ra = GlobalRelationAnalyzer(ppo=self.ppo_g, path=pp.se.path, execution=exe,
                                    solver=pp.se.solver, litmus=litmus)

        # set the color for each relation.
        colors = {
            'po': 'black',
            'co': 'blue',
            'rf': 'red',
            'ppo': '#802E89',
            'fr': '#ffcc33',
            'rmw': '#993c28'
        }

        def add_edges(name):
            for e1, e2 in ra.find_all(name):
                edges.append(AGraphEdge(src=nodes[e1].name, tgt=nodes[e2].name, label=name, color=colors[name]))

        # config edges displayed in the graph.
        edge_displayed = [
            # 'po',
            'co',
            'rf',
            'fr',
            'ppo',  # TODO: annotate the specific ppo rule.
            'rmw',
        ]
        for r in edge_displayed:
            add_edges(r)

        clusters = {}
        for e in exe:
            cluster_name = 'Init' if e.pid == -1 else f'Thread {e.pid}'
            cluster = clusters.setdefault(e.pid, Cluster(label=cluster_name))
            cluster.nodes.append(str(e))

        plot_graph(filename, list(nodes.values()), edges, list(clusters.values()))
        INFO(filename, f'plot execution {exe} of {litmus.name}')
