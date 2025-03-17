
"""RISC-V TSO Memory Model"""
from src.litmusbayes.analysis import Event
from src.litmusbayes.analysis.model.C910 import C910
from src.litmusbayes.analysis.model.rvwmo import rvwmo_global_ppos, rvwmo_local_ppos
from src.litmusbayes.analysis.rel import PPO




def ppo_C910_4(ra, e1: Event, e2: Event) -> bool:
    return ra.AMO(e1) and ra.po(e1,e2) and ra.AMO(e2)


def ppo_C910_5(ra, e1: Event, e2: Event) -> bool:
    return ra.R(e1) and ra.po(e1,e2) and ra.AMO(e2)


def ppo_C910_6(ra, e1: Event, e2: Event) -> bool:
    return ra.W(e1) and ra.po(e1,e2) and ra.AMO(e2)


def ppo_C910_7(ra, e1: Event, e2: Event) -> bool:
    return ra.AMO(e1) and ra.po(e1,e2) and ra.R(e2)


def ppo_C910_8(ra, e1: Event, e2: Event) -> bool:
    return ra.AMO(e1) and ra.po(e1,e2) and ra.W(e2)

class C910_AMO(C910):
    def __init__(self, plot_enabled: bool = False):
        super().__init__(plot_enabled=plot_enabled)
        self.ppo_g = PPO(rvwmo_global_ppos + [ppo_C910_4,ppo_C910_5,ppo_C910_6,ppo_C910_7,ppo_C910_8])
        self.ppo_l = PPO(rvwmo_local_ppos + [ppo_C910_4,ppo_C910_5,ppo_C910_6,ppo_C910_7,ppo_C910_8])
