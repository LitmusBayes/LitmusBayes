

"""RISC-V TSO Memory Model"""
from src.litmusbayes.analysis import Event
from src.litmusbayes.analysis.model.rvwmo import RVWMO, rvwmo_global_ppos, rvwmo_local_ppos
from src.litmusbayes.analysis.rel import PPO



def ppo_C910_1(ra, e1: Event, e3) -> bool:
    return any([ra.R(e1) and ra.po(e1,e2) and ra.R(e2) and ra.rmw(e2,e3) and ra.X(e3)  for e2 in ra.execution ])

def ppo_C910_2(ra, e1: Event, e3) -> bool:
    return any([ra.R(e1) and ra.rmw(e1,e2) and ra.X(e2) and ra.po(e2,e3) and ra.W(e3)  for e2 in ra.execution ])

def ppo_C910_3(ra, e1: Event, e4) -> bool:
    return any([ra.R(e1) and ra.rmw(e1,e2) and ra.X(e2) and ra.po(e2,e3) and ra.R(e3) and ra.rmw(e3,e4) and ra.X(e4)  for e2 in ra.execution  for e3 in ra.execution ])


class C910(RVWMO):
    def __init__(self, plot_enabled: bool = False):
        super().__init__(plot_enabled=plot_enabled)
        self.ppo_g = PPO(rvwmo_global_ppos + [ppo_C910_1, ppo_C910_2, ppo_C910_3])
        self.ppo_l = PPO(rvwmo_local_ppos + [ppo_C910_1, ppo_C910_2, ppo_C910_3])
