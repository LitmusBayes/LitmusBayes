
"""RISC-V Weak Memory Ordering"""
from src.litmusbayes.analysis import Event
from src.litmusbayes.analysis.model.mm import MemoryModel
from src.litmusbayes.analysis.rel import PPO
from src.litmusbayes.prog import *


def ppo_r1(ra, e1: Event, e2: Event) -> bool:
    """
    ppo rule 1: Rule 1: b is a store, and a and b access overlapping memory addresses
    """
    return ra.po(e1, e2) and e2.type == EType.Write and ra.loc(e1, e2)


def ppo_r4(ra, e1: Event, e2: Event) -> bool:
    return ra.fence(e1, e2)


def ppo_r5(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and isinstance(e1.inst.inst, AmoInst) and \
        e1.inst.inst.flag in [MoFlag.Strong, MoFlag.Acquire]


def ppo_r6(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and isinstance(e2.inst.inst, AmoInst) and \
        e2.inst.inst.flag in [MoFlag.Strong, MoFlag.Release]


def ppo_r7(ra, e1: Event, e2: Event) -> bool:
    RCsc = lambda e: isinstance(e.inst.inst, AmoInst) and e.inst.inst.flag != MoFlag.Relax
    return ra.po(e1, e2) and RCsc(e1) and RCsc(e2)


def ppo_r8(ra, e1: Event, e2: Event) -> bool:
    # TODO: (deheng) while ppo8 is rwm and the rmw() is defined, the code is not consistent with
    #  rmw()
    if not ra.po(e1, e2):
        return False
    if not ra.R(e1) or not ra.W(e2):
        return False

    def is_inst_type(e: Event, t: IType):
        if not e.inst:
            return False
        return e.inst.inst.type == t

    if not (is_inst_type(e1, IType.Lr) and is_inst_type(e2, IType.Sc) and ra.loc(e1, e2)):
        return False

    lr_sc_between = [e for e in ra.execution
                     if ra.po(e1, e) and ra.po(e, e2)
                     and is_inst_type(e, IType.Lr) or is_inst_type(e, IType.Sc)]
    return len(lr_sc_between) == 0


def ppo_r9(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.addr(e1, e2)


def ppo_r10(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.data(e1, e2)


def ppo_r11(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.W(e2) and ra.ctrl(e1, e2)


def ppo_r13(ra, e1: Event, e2: Event) -> bool:
    """
    Rule 13: b is a store, and there exists some instruction m between a and b in program order
    such that m has an address dependency on a.
    In ISA P100, if a has addr/data/ctrl dependency on b, a and b must be memory operations.
    SC_Fail instruction has no dependency on any other instructions.
    """
    events = ra.execution
    is_m = lambda e: ra.po(e1, e) and ra.po(e, e2) and ra.addr(e1, e)
    return e2.type == EType.Write and any(is_m(e) for e in events if e.type is not EType.SC_Fail)



def ppo_r3(ra, e1: Event, e2: Event) -> bool:
    if not e1.inst or not e2.inst:
        return False
    return (e1.inst.type == IType.Sc or e1.inst.type == IType.Amo) and ra.rfi(e1, e2)


def ppo_r12(ra, e1: Event, e2: Event) -> bool:
    """
    Rule 12: b is a load, and there exists some store m between a and b in program order such that
    m has an address or data dependency on a, and b returns a value written by m.

    1) get the store m
    2) check there is no store n between m and b such that n, b operate on the same address
    """

    if not ra.po(e1, e2):
        return False

    events = ra.execution
    is_m = lambda e: ra.po(e1, e) and ra.rfi(e, e2) and (ra.addr(e1, e) or ra.data(e1, e))
    return any(is_m(e) for e in events)


rvwmo_global_ppos = [ppo_r1, ppo_r3, ppo_r4, ppo_r5, ppo_r6, ppo_r7,
                     ppo_r8, ppo_r9,ppo_r10, ppo_r11, ppo_r12, ppo_r13]
rvwmo_local_ppos = [p for p in rvwmo_global_ppos if p not in [ppo_r3, ppo_r12]]


class RVWMO_WITHOUT_PPO2(MemoryModel):
    def __init__(self, plot_enabled: bool = False):
        super().__init__(ppo_g=PPO(rvwmo_global_ppos),
                         ppo_l=PPO(rvwmo_local_ppos),
                         plot_enabled=plot_enabled)
