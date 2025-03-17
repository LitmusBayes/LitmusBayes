

"""RISC-V TSO Memory Model"""
from src.litmusbayes.analysis import Event
from src.litmusbayes.analysis.model.rvwmo import RVWMO, rvwmo_global_ppos, rvwmo_local_ppos
from src.litmusbayes.analysis.rel import PPO

def ppo_X_1(ra, e1: Event, e2) -> bool:
    return ra.R(e1) and ra.po(e1,e2) and ra.X(e2)



class RVWMOX(RVWMO):
    def __init__(self, plot_enabled: bool = False):
        super().__init__(plot_enabled=plot_enabled)
        self.ppo_g = PPO(rvwmo_global_ppos + [ppo_X_1])
        self.ppo_l = PPO(rvwmo_local_ppos + [ppo_X_1])
