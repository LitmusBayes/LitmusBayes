
"""RISC-V TSO Memory Model"""
from src.litmusbayes.analysis import Event
from src.litmusbayes.analysis.model.rvwmo import RVWMO, rvwmo_global_ppos, rvwmo_local_ppos
from src.litmusbayes.analysis.rel import PPO


def ppo_r14(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.R(e1)


def ppo_r15(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.W(e2)


class RVTSO(RVWMO):
    def __init__(self, plot_enabled: bool = False):
        super().__init__(plot_enabled=plot_enabled)
        self.ppo_g = PPO(rvwmo_global_ppos + [ppo_r14, ppo_r15])
        self.ppo_l = PPO(rvwmo_local_ppos + [ppo_r14, ppo_r15])
