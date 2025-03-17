

"""RISC-V Sequential Consistency Memory Model"""
from src.slide.analysis import Event
from src.slide.analysis.model.mm import MemoryModel
from src.slide.analysis.rel import PPO


def ppo_r1(ra, e1: Event, e2: Event) -> bool:
    return ra.po(e1, e2)


class SC(MemoryModel):
    def __init__(self, plot_enabled: bool = False):
        super().__init__(ppo_g=PPO(
            [ppo_r1]),
            ppo_l=PPO([ppo_r1]),
            plot_enabled=plot_enabled)
