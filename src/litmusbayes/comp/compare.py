

from src.litmusbayes.litmus.litmus import LitmusResult
from src.litmusbayes.utils import list_util


def compare_two_results(rs1: LitmusResult, rs2: LitmusResult):
    """
    :param rs1:
    :param rs2:
    :return:
    """
    unique_states_in_rs1 = []
    unique_states_in_rs2 = []
    is_same = True
    if not list_util.is_same(rs1.states, rs2.states):
        is_same = False
        unique_states_in_rs1 = list_util.get_uniq_in_src(rs1.states, rs2.states)
        unique_states_in_rs2 = list_util.get_uniq_in_src(rs2.states, rs1.states)
    return is_same, unique_states_in_rs1, unique_states_in_rs2

