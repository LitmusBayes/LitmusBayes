
"""Extract Relations from Events"""
from typing import List

import z3

from src.litmusbayes.analysis.event import Event
from src.litmusbayes.litmus import Litmus
from src.litmusbayes.prog import EType, SSAInst, AmoInst, MoFlag, IType



def PPO(rules: list = None):
    rules = rules if rules else []

    def wrapper(ra: LocalRelationAnalyzer, e1, e2):
        if not ra.po(e1, e2):
            return False

        if e1.type is EType.SC_Fail or e2.type is EType.SC_Fail:
            return False
        
        for ppo_rule in rules:
            # print(ppo_rule,e1,e2,ra)
            # if(ppo_rule(ra, e1, e2)):
            #    print('ppo_rule true',ppo_rule,e1,e2)
            if ppo_rule(ra, e1, e2):
                return True
        return False

    return wrapper


def subpath(path, start: SSAInst, end: SSAInst) -> list:
    if start not in path or end not in path:
        return []
    return path[path.index(start):path.index(end)]


class LocalRelationAnalyzer:
    """Local Relation Analyzer
    Local relations refer to those that can be determined within an execution path in one hart,
    """

    def __init__(self, ppo=None,
                 path: List[SSAInst] = None,
                 execution: List[Event] = None,
                 solver: z3.Solver = None,
                 litmus: Litmus = None):

        self._ppo = ppo
        self.litmus = litmus
        self.path = path
        self.execution = execution
        self.solver = solver

        self._locs = {}
        self.loc_val = litmus.loc_val.copy() if litmus else {}
        self._addrs, self._datas, self._ctrls = {}, {}, {}
        self._ppos = {}

    def clear(self):
        # keep 'loc' and clear 'not loc'
        loc_keep = [k for k, v in self._locs.items() if v]
        self._locs = {k: v for k, v in self._locs.items() if k in loc_keep}

        self._addrs.clear()
        self._datas.clear()
        self._ctrls.clear()
        self._ppos.clear()

    def set_ppo_rules(self, ppo_rules):
        """Reset ppo rules."""
        self._ppo = ppo_rules
        self._ppos.clear()

    def check(self, cond) -> bool:
        return self.solver.check(cond) == z3.sat

    def R(self, e1: Event, e2: Event = None) -> bool:
        assert e1 in self.execution
        if e2 is not None:
            assert e1 is e2
        return e1.type is EType.Read

    def W(self, e1: Event, e2: Event = None) -> bool:
        assert e1 in self.execution
        if e2 is not None:
            assert e1 is e2
        return e1.type is EType.Write

    def M(self, e1: Event, e2: Event = None) -> bool:
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return True

    def AMO(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return e1.inst.type == IType.Amo

    def X(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return e1.inst.type == IType.Sc or e1.inst.type == IType.Lr

    def XSc(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return e1.inst.type == IType.Sc

    def XLr(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return e1.inst.type == IType.Lr

    def RCsc(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return isinstance(e1.inst.inst, AmoInst) and e1.inst.inst.flag != MoFlag.Relax

    def AQ(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return isinstance(e1.inst.inst, AmoInst) and e1.inst.inst.flag in [MoFlag.Strong, MoFlag.Acquire]

    def RL(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return isinstance(e1.inst.inst, AmoInst) and e1.inst.inst.flag in [MoFlag.Strong, MoFlag.Release]
    
    def AQRL(self, e1: Event, e2: Event = None):
        assert e1 in self.execution and (e1 is e2 if e2 else True)
        return isinstance(e1.inst.inst, AmoInst) and e1.inst.inst.flag in [MoFlag.Strong]

    def po(self, e1: Event, e2: Event) -> bool:
        assert e1 in self.execution and e2 in self.execution
        return e1.pid == e2.pid and ((e1.idx < e2.idx) or (e1.idx == e2.idx and e1.inst == e2.inst and e1.type == EType.Read and e2.type == EType.Write))


    def po_loc(self, e1: Event, e2: Event) -> bool:
        return self.po(e1, e2) and self.loc(e1, e2)

    def find_all_by_func(self, func) -> List[tuple[Event, Event]]:
        relation_list = []
        for e1 in self.execution:
            for e2 in self.execution:
                if e1.inst == None or e2.inst == None:
                    continue
                if func(self, e1, e2):
                    relation_list.append((e1, e2))
        return relation_list
        # return [(e1, e2)
        #         for e1 in self.execution
        #         for e2 in self.execution
        #         if func(self, e1, e2)]

    def find_all(self, r) -> List[tuple[Event, Event]]:
        if r is not 'co':
            return [(e1, e2)
                    for e1 in self.execution
                    for e2 in self.execution
                    if getattr(self, r)(e1, e2)]
        else:
            # for co
            rels = []
            writes = [e for e in self.execution if self.W(e)]
            for i in range(len(writes)):
                for j in range(i, len(writes)):
                    e1, e2 = writes[i], writes[j]
                    if getattr(self, r)(e1, e2):
                        rels.append((e1, e2))
                        break
            return rels

    def ppo(self, e1: Event, e2: Event):
        if (e1, e2) in self._ppos:
            return self._ppos[(e1, e2)]
        if not self._ppo:
            raise NotImplementedError
        is_ppo = self._ppo(self, e1, e2)
        self._ppos[(e1, e2)] = is_ppo
        return is_ppo

    # @profile
    def loc(self, e1: Event, e2: Event) -> bool:
        """
        Check if e1 and e2 have overlapped address
        """
        if e1 is e2:
            return False
        if e1.addr is None or e2.addr is None:
            return False
        if (e1, e2) in self._locs:
            return self._locs[(e1, e2)]
        if e1 in self.loc_val and e2 in self.loc_val:
            e1_v, e2_v = self.loc_val[e1], self.loc_val[e2]
            if e1_v == e2_v:
                is_loc = True
            else:
                is_loc = not self.check(e1.addr != e2.addr)
                if is_loc:
                    for k in self.loc_val:
                        if self.loc_val[k] == e2_v:
                            self.loc_val[k] = e1_v
                else:
                    for a in [k for k, v in self.loc_val.items() if v == e1_v]:
                        for b in [k for k, v in self.loc_val.items() if v == e2_v]:
                            self._locs[(a, b)], self._locs[(b, a)] = False, False
        else:
            is_loc = not self.check(e1.addr != e2.addr)
            if is_loc:
                if e1 in self.loc_val:
                    self.loc_val[e2] = self.loc_val[e1]
                elif e2 in self.loc_val:
                    self.loc_val[e1] = self.loc_val[e2]
                else:
                    val = len(self.loc_val.keys())
                    self.loc_val[e1] = val
                    self.loc_val[e2] = val
        self._locs[(e1, e2)] = is_loc
        self._locs[(e2, e1)] = is_loc
        return is_loc

    def _dep(self, e1: Event, e2: Event, sinks: list) -> bool:
        """Check the dependency (data, addr and ctrl)
        between e1 and e2 from a set of sink variables."""
        if len(sinks) == 0 or None in sinks:
            return False

        if not self.po(e1, e2):
            return False

        path = self.path
        worklist = [i for i in subpath(path, e1.inst, e2.inst) if i.get_def() in sinks]
        while len(worklist) > 0:
            # traverse the execution path with BFS.
            if e1.inst in worklist:
                return True
            current: SSAInst = worklist.pop(0)

            for u in current.get_uses():
                # add all insts to worklist along the def-use chain.
                worklist.extend([i for i in subpath(path, e1.inst, current) if u == i.get_def()])

        return False

    def addr(self, e1: Event, e2: Event) -> bool:
        if (e1, e2) in self._addrs:
            return self._addrs[(e1, e2)]
        if not self.po(e1, e2):  # need to do the check, otherwise e2.inst may be None
            return False
        is_addr = self._dep(e1, e2, [e2.inst.get_addr()])
        self._addrs[(e1, e2)] = is_addr
        return is_addr

    def data(self, e1: Event, e2: Event) -> bool:
        if (e1, e2) in self._datas:
            return self._datas[(e1, e2)]
        if not self.po(e1, e2):
            return False

        is_data = self._dep(e1, e2, [e2.inst.get_data()])
        self._datas[(e1, e2)] = is_data
        return is_data

    def ctrl(self, e1: Event, e2: Event) -> bool:
        if not self.po(e1, e2):
            return False

        if (e1, e2) in self._ctrls:
            return self._ctrls[(e1, e2)]

        path = self.path
        sinks = []
        for i in subpath(path, e1.inst, e2.inst):
            if i.branch_taken is not None:
                sinks.extend(i.get_uses())

        is_ctrl = self._dep(e1, e2, list(set(sinks)))
        self._ctrls[(e1, e2)] = is_ctrl
        return is_ctrl

    def fence(self, e1: Event, e2: Event) -> bool:
        if not self.po(e1, e2):
            return False

        path = self.path
        fences = list(
            filter(lambda i: i.type == IType.Fence or i.type == IType.FenceTso,
                   subpath(path, e1.inst, e2.inst)))
        return any(map(lambda f: f.inst.ordered(e1.type, e2.type), fences))

    def _fence_sync(self, e1: Event, e2: Event, fence_mode ) -> bool: #('rw','rw')
        if not self.po(e1, e2):
            return False
        path = self.path
        fence_type_list = [('rw', 'rw'),
                           ('rw', 'r'),
                           ('rw', 'w'),
                           ('r',  'rw'),
                           ('r',  'r'),
                           ('r',  'w'),
                           ('w',  'rw'),
                           ('w',  'r'),
                           ('w',  'w'),
                           ]
        fences = []
        if fence_mode in fence_type_list:
            fences = list(filter(lambda i: i.type == IType.Fence and i.inst.pre==fence_mode[0] and i.inst.suc==fence_mode[1], subpath(path, e1.inst, e2.inst)))
        elif fence_mode == ('tso', ''):
            fences = list(filter(lambda i: i.type == IType.FenceTso, subpath(path, e1.inst, e2.inst)))
        else:
            assert False, 'fence_mode is error'
        return len(fences) > 0

    def fence_rw_rw(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('rw', 'rw'))

    def fence_rw_w(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('rw', 'w'))

    def fence_rw_r(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('rw', 'r'))

    def fence_r_rw(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('r', 'rw'))

    def fence_r_w(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('r', 'w'))

    def fence_r_r(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('r', 'r'))

    def fence_w_rw(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('w', 'rw'))

    def fence_w_r(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('w', 'r'))

    def fence_w_w(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('w', 'w'))

    def fence_tso(self, e1: Event, e2: Event) -> bool:
        return self._fence_sync(e1, e2, ('tso', ''))


    def amo(self, e1: Event, e2: Event) -> bool:
        """e1 and e2 are from the same amo inst."""
        if not e1.inst or not e2.inst:
            return False

        if e1 is e2:
            return False

        return e1.inst is e2.inst and e1.is_amo() and e2.is_amo()

    def rmw(self, e1: Event, e2: Event) -> bool:
        """a read event e1 and a write event e2 are from the same amo inst or paired lr & sc."""

        if e1 is e2:
            return False

        if not e1.inst or not e2.inst:
            return False

        if not (self.R(e1) and self.W(e2)):
            return False

        # lr & sc
        if e1.inst.name.startswith('lr') and \
                e2.inst.name.startswith('sc') and \
                self.po(e1, e2) and \
                self.loc(e1, e2):
            for e in [e for e in self.execution if e.inst and e.inst.name.startswith('lr')]:
                # there is no another lr between them
                if self.po(e, e2) and self.po(e1, e) and self.loc(e, e2):
                    return False
            return True

        # amo
        if e1.inst is e2.inst:
            return True

        return False


class GlobalRelationAnalyzer(LocalRelationAnalyzer):
    def __init__(self, ppo=None, path: List[SSAInst] = None,
                 execution: List[Event] = None,
                 solver: z3.Solver = None,
                 litmus: Litmus = None):
        super().__init__(ppo, path, execution, solver, litmus)
        self._rfs, self._frs = {}, {}
        self.exe_value_map = {}

    def clear(self):
        super().clear()
        self._rfs.clear()
        self._frs.clear()

    def int(self, e1: Event, e2: Event) -> bool:
        assert e1 in self.execution and e2 in self.execution
        return e1.pid == e2.pid

    def ext(self, e1: Event, e2: Event) -> bool:
        assert e1 in self.execution and e2 in self.execution
        return e1.pid != e2.pid

    def rsw(self, e1: Event, e2: Event) -> bool: # Question: need int?
        return any(self.rf(e, e1) and self.rf(e, e2) for e in self.execution)

    def rfi(self, e1: Event, e2: Event) -> bool:
        return self.int(e1, e2) and self.rf(e1, e2)

    def rfe(self, e1: Event, e2: Event) -> bool:
        return self.ext(e1, e2) and self.rf(e1, e2)

    def fri(self, e1: Event, e2: Event) -> bool:
        return self.int(e1, e2) and self.fr(e1, e2)

    def fre(self, e1: Event, e2: Event) -> bool:
        return self.ext(e1, e2) and self.fr(e1, e2)

    def coi(self, e1: Event, e2: Event) -> bool:
        return self.int(e1, e2) and self.co(e1, e2)

    def coe(self, e1: Event, e2: Event) -> bool:
        return self.ext(e1, e2) and self.co(e1, e2)

    def gmo(self, e1: Event, e2: Event) -> bool:
        return self.execution.index(e1) < self.execution.index(e2)

    # @profile
    def co(self, e1: Event, e2: Event) -> bool:
        return e2.pid >= 0 and \
            self.gmo(e1, e2) and \
            e1.type is EType.Write and \
            e2.type is EType.Write and \
            self.loc(e1, e2)

    # @profile
    def rf(self, e1: Event, e2: Event) -> bool:
        """ Check if e1 and e2 satisfy read-from (rf) relation.
        See 'load value axiom' for details."""
        if e1.type is not EType.Write or e2.type is not EType.Read:
            return False
        if (e1, e2) in self._rfs:
            return self._rfs[(e1, e2)]
        candidates = [e for e in self.execution if
                      e.type is EType.Write and (self.po(e, e2) or self.gmo(e, e2))]
        candidates.reverse()
        is_rf = False
        for e in candidates:
            if self.loc(e, e2):
                is_rf = e is e1
                break

        self._rfs[(e1, e2)] = is_rf
        return is_rf

    def fr(self, e1: Event, e2: Event) -> bool:
        if (e1, e2) in self._frs:
            return self._frs[(e1, e2)]
        is_fr = False
        for e in self.execution:
            if self.rf(e, e1):
                if self.co(e, e2):
                    is_fr = True
                break

        self._frs[(e1, e2)] = is_fr
        return is_fr