import os
import sys

sys.path.append("../../../src")
from src.litmusbayes.analysis.model import RVWMO
from src.litmusbayes.comp.parse_result import parse_chip_log, parse_herd_log
from src.litmusbayes.litmus.strategy.delay_strategy import DelayStrategy
from src.litmusbayes.litmus.strategy.fence_strategy import FenceStrategy
from src.litmusbayes.litmus.litmus_changer import InjectList
from src.litmusbayes.prog.inst import *
from src.litmusbayes.litmus import parse_litmus
from src.litmusbayes.utils.file_util import *
from src.litmusbayes import config
import time

litmus_log_cnt_dir = {}
litmus_log_cnt_path = os.path.join(config.LITMUS_TRANS_DIR_PATH, 'litmus_trans_log_cnt.log')
if not os.path.exists(litmus_log_cnt_path):
   with open(litmus_log_cnt_path, 'w') as f:
       f.write('')
else:
    with open(litmus_log_cnt_path, 'r') as f:
        for line in f:
            litmus_name, cnt = line.strip().split('|')[0],int(line.strip().split('|')[1])
            if litmus_name in litmus_log_cnt_dir:
                litmus_log_cnt_dir[litmus_name] = max(litmus_log_cnt_dir[litmus_name],cnt)
            else:
                litmus_log_cnt_dir[litmus_name] = cnt

class TransMode:
    DIFF = 0
    DIRECT = 1

class LitmusTransformer:
    def __init__(self):
        self.dif_litmus = None
        self.dif_dict = None
        self.litmus_name = ''
        self.strategy_list = [
            FenceStrategy(),
            DelayStrategy(),
        ]
        self.cnt = 0

    def clear(self):
        self.dif_litmus = None
        self.dif_dict = None
        self.litmus_name = ''
        self.cnt = 0

    def set(self, litmus_name, dif_litmus, dif_dict, cnt):
        self.litmus_name = litmus_name
        self.dif_litmus = dif_litmus
        self.dif_dict = dif_dict
        self.cnt = cnt

    def get_litmus_path(self):
        litmus_path = os.path.join(config.LITMUS_TRANS_DIR_PATH,f'{self.litmus_name}_{self.cnt}.litmus')
        self.cnt += 1
        return litmus_path


    def print_init_states(self):
        for state in self.dif_dict:
            print(state)
            exe = self.dif_dict[state]['exe']
            print('exe')
            for e in exe:
                print(e)


    def transform_by_one(self, state, exe, ra):
        mutate_litmus_list = []
        for strategy in self.strategy_list:
            strategy.clear()
            strategy.set(self.dif_litmus, state, exe, ra, self.get_litmus_path())
            mutate_litmus, inject_list = strategy.litmus_transform()
            if len(inject_list.inject_list) ==0 :
                continue
            mutate_litmus_list.append((mutate_litmus, inject_list))
            print('mutate_litmus:', mutate_litmus)

        return mutate_litmus_list


    def transform(self):
        assert self.dif_litmus is not None
        assert self.dif_dict is not None

        mutate_array = []
        inject_list_statistics = []
        for state in self.dif_dict:
            exe = self.dif_dict[state]['exe']
            ra = self.dif_dict[state]['ra']
            mutate_litmus_list =self.transform_by_one(state, exe, ra)
            for mutate_litmus, inject_list in mutate_litmus_list:
                if inject_list is None:
                    continue
                flag = True
                for inject_list_tmp in inject_list_statistics:
                    if inject_list == inject_list_tmp:
                        flag = False
                        break
                if flag:
                    mutate_array.append((mutate_litmus, exe))
                    inject_list_statistics.append(inject_list)

        return mutate_array



def get_litmus_state_dif(litmus, chip_states):
    config.init()
    config.set_var('reg_size', 64)

    mm = RVWMO()
    mm.run(litmus)
    mm_states = mm.states
    dif_state_set = set(mm_states)-set(chip_states)
    dif_dict = {}
    for state in dif_state_set:
        exe, ra = mm.find_exe_by_state(state), mm.find_ra_by_state(state)
        dif_dict[state] = {'exe': exe, 'ra': ra}

    return dif_dict

def get_litmus_state_direct(litmus, chip_states):
    config.init()
    config.set_var('reg_size', 64)

    mm = RVWMO()
    mm.run(litmus)
    direct_dict = {}
    for state in chip_states:
        exe, ra = mm.find_exe_by_state(state), mm.find_ra_by_state(state)
        direct_dict[state] = {'exe': exe, 'ra': ra}

    return direct_dict


def get_transform_litmus_list(litmus_name, litmus_path, chip_states, mode = TransMode.DIFF):
    litmus_transformer = LitmusTransformer()

    litmus = parse_litmus(read_file(litmus_path))
    # print(read_file(litmus_path))
    state_dif_dict = {}
    if mode == TransMode.DIFF:
        state_dif_dict = get_litmus_state_dif(litmus, chip_states)
    elif mode == TransMode.DIRECT:
        state_dif_dict = get_litmus_state_direct(litmus, chip_states)
    #litmus transform
    litmus_log_cnt_dir.setdefault(litmus_name, 0)
    litmus_transformer.clear()
    litmus_transformer.set(litmus_name, litmus, state_dif_dict, litmus_log_cnt_dir[litmus_name])
    # litmus_transformer.print_init_states()

    mutate_list = litmus_transformer.transform()
    new_litmus_list = []
    for litmus, exe in mutate_list:
        new_litmus_path = os.path.join(config.LITMUS_TRANS_DIR_PATH,f'{litmus_name}_{litmus_log_cnt_dir[litmus_name]}.litmus')
        litmus.mutate_new_litmus(InjectList(), new_litmus_path)
        new_litmus_list.append(new_litmus_path)
        litmus_log_cnt_dir[litmus_name] += 1
        write_line_to_file(litmus_log_cnt_path, f'{litmus_name}|{litmus_log_cnt_dir[litmus_name]}')

    # for path in new_litmus_list:
    #     print(path)

    return new_litmus_list



if __name__ == '__main__':
    litmus_name = 'ISA12'
    litmus_path = search_file(litmus_name,f'{config.INPUT_DIR}/litmus', '.litmus')
    states = {r.name: r.states for r in parse_herd_log(os.path.join(config.INPUT_DIR, 'herd/herd_results_rvwmo.log'))}[litmus_name]
    print(states)
    get_transform_litmus_list(litmus_name, litmus_path, states, mode=TransMode.DIRECT)