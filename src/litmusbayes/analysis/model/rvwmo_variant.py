from src.litmusbayes.analysis import Event
from src.litmusbayes.analysis.model.rvwmo import RVWMO, rvwmo_global_ppos, rvwmo_local_ppos
from src.litmusbayes.analysis.rel import PPO
from src.litmusbayes.prog import EType


def ppo_RW(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.R(e1) and ra.W(e2)

def ppo_AMO_X(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and (
        ((ra.X(e1) or ra.AMO(e1)) and ra.M(e2)) or
        ((ra.X(e2) or ra.AMO(e2)) and ra.M(e1)))

def ppo_fence(ra, e1: Event, e2: Event) -> bool:
    return (ra.po(e1, e2) and
            (ra.fence_rw_rw(e1, e2) or ra.fence_rw_r(e1, e2) or
             ra.fence_rw_w(e1, e2) or ra.fence_w_rw(e1, e2) or
             ra.fence_w_r(e1, e2) or ra.fence_w_w(e1, e2) or
             ra.fence_r_rw(e1, e2) or ra.fence_r_r(e1, e2) or
             ra.fence_r_w(e1, e2) or ra.fence_tso(e1, e2)
             ) and ra.M(e1) and ra.M(e2))

def ppo_SC(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ra.M(e1) and ra.M(e2)

def ppo_TSO(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2) and ((ra.R(e1) and ra.M(e2)) or (ra.W(e1) and ra.W(e2)))


def ppo_strong_ppo12(ra, e1: Event, e2: Event) -> bool:

    if not ra.po(e1, e2):
        return False

    events = ra.execution
    is_m = lambda e: ra.po(e1, e) and ra.po(e, e2) and ra.R(e2) and (ra.addr(e1, e) or ra.data(e1, e))
    return any(is_m(e) for e in events)



def ppo_strong_ppo13(ra, e1: Event, e2: Event) -> bool:
    events = ra.execution
    is_m = lambda e: ra.po(e1, e) and ra.po(e, e2) and ra.addr(e1, e)
    return e2.type == EType.Read and any(is_m(e) for e in events if e.type is not EType.SC_Fail)


ppo_dict = {
    'RW': [ppo_RW],
    'AMO_X': [ppo_AMO_X],
    'fence': [ppo_fence],
    'SC': [ppo_SC],
    'TSO': [ppo_TSO],
    'strong_ppo12': [ppo_strong_ppo12],
    'strong_ppo13': [ppo_strong_ppo13]
}

class RVWMO_variant(RVWMO):
    def __init__(self, plot_enabled: bool = False, variant_array = []):
        super().__init__(plot_enabled=plot_enabled)
        ppo_g = rvwmo_global_ppos
        ppo_l = rvwmo_local_ppos
        for variant_name in variant_array:
            ppo_l.extend(ppo_dict[variant_name])
            ppo_g.extend(ppo_dict[variant_name])
        self.ppo_g = PPO(ppo_g)
        self.ppo_l = PPO(ppo_l)
