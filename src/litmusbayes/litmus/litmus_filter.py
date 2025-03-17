import os

from src.litmusbayes import config
from src.litmusbayes.litmus import parse_litmus, Litmus
from src.litmusbayes.litmus.load_litmus import get_litmus_by_policy, GetLitmusPolicy
from src.litmusbayes.prog import AmoInst, IType
from src.litmusbayes.utils.file_util import read_file, get_file_name_without_suffix_by_path


class LitmusFilter:
    def __init__(self, litmus_suite):
        self.litmus_suite = litmus_suite

    def filter(self, func, output_path):
        filter_suite = [litmus for litmus in self.litmus_suite if func(parse_litmus(read_file(litmus)))]
        with open(output_path, 'w') as f:
            for litmus in filter_suite:
                f.write(get_file_name_without_suffix_by_path(litmus))
                f.write('\n')
        return filter_suite

def filter_X_thread_pass_two(litmus: Litmus):
    X_thread_num = 0
    for prog in litmus.progs:
        for inst in prog.insts:
            if type(inst) == AmoInst and inst.type in [IType.Lr,IType.Sc]:
                X_thread_num += 1
                break
    if X_thread_num >= 2:
        return True
    else:
        return False

def filter_X_exceed_four(litmus: Litmus):
    X_num = 0
    print(litmus.name)
    for prog in litmus.progs:
        for inst in prog.insts:
            if type(inst) == AmoInst and inst.type in [IType.Lr,IType.Sc]:
                X_num += 1
                print(inst)
    return X_num >= 4

if __name__ == '__main__':

    litmus_files = get_litmus_by_policy(GetLitmusPolicy.All, None)
    print(1)
    # print(litmus_files)
    litmus_filter = LitmusFilter(litmus_files)
    # path = os.path.join(config.INPUT_DIR, 'chip_execution_logs/exceed_two_threads_has_X.txt')
    # filter_litmus_suite = litmus_filter.filter(filter_X_thread_pass_two, path)
    path = os.path.join(config.INPUT_DIR, 'chip_execution_logs/X_exceed_four.txt')
    filter_litmus_suite = litmus_filter.filter(filter_X_exceed_four, path)
    for litmus in filter_litmus_suite:
        print(litmus)