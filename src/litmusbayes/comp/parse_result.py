

import re
from typing import List

from src.litmusbayes.litmus.litmus import LitmusResult
from src.litmusbayes.log import WARNING
from src.litmusbayes.utils import regex_util, file_util


def parse_one_herd_output(output):
    results = re.findall(re.compile(r"States (.*?)\n.*?Positive: (.*?) Negative: (.*?)\n.*?Time .*? (.*?)\n",
                                    re.S), output)
    # assert len(results) == 1, f"Error: {output}"
    if len(results)==0:
        return []
    result = results[0]
    states_cnt = int(result[0])
    pos_cnt = int(result[1])
    neg_cnt = int(result[2])
    time_cost = float(result[3])

    # parse states list.
    # 0:x7=0; 0:x8=0; 1:x7=0; 1:x8=1; [x]=2;
    # [x]=1; [y]=1;
    """e.g., Observation 2+2W+fence.rw.ws Never 0 3
    States 3
    [x]=1; [y]=1;
    [x]=1; [y]=2;
    [x]=2; [y]=1;
    No
    Witnesses
    """
    # TODO: use pattern like [Ok|No]
    states_str = re.findall(re.compile(r"\nStates .*?\n(.*)\nOk\n", re.S), output)
    if len(states_str) == 0:
        states_str = re.findall(re.compile(r"\nStates .*?\n(.*)\nNo\n", re.S), output)
    if len(states_str) == 0:
        WARNING(f'current output has no states: {output}.')
        return [{}], None, None, None
    # states_str_list = states_str[0].replace("[", "").replace("]", "").split("\n")
    states_str_list = states_str[0].split("\n")
    assert len(
        states_str_list) == states_cnt, f'[ERROR] [states cnt check fails] states: {states_str}; states_cnt: {states_cnt}.'
    state_dict_list = get_state_dict_list_from_str(states_str_list)
    return state_dict_list, pos_cnt, neg_cnt, time_cost


def get_state_dict_list_from_str(states_str):
    state_dict_list = []  # the element of the list is a state dict
    for i, state in enumerate(states_str):
        state_dict = {}
        for equation in [eq.strip() for eq in state.split(';') if eq != '']:
            if equation == '':
                continue
            key, value = equation.split('=')
            if len(key) == 1:  # is variable,e.g., x y z
                # assert key in ['x', 'y', 'z', 'a', 'b', 'c'], f'[ERROR] unknown key: {key}'
                key = '[' + key + ']'  # this is for chip state
            assert key not in state_dict.keys()
            state_dict[key] = value
        state_dict_list.append(state_dict)
    return state_dict_list


def parse_herd_log(filepath) -> List[LitmusResult]:
    """
    Parse a log file generated from herd.
    :param filepath: the path of log file
    :return: a list of LitmusResult
    """
    from src.litmusbayes.utils.file_util import read_file
    from src.litmusbayes.litmus import LitmusState

    content = read_file(filepath)
    outputs = [output for output in content.split('\n\n') if len(output) > 0 and not output.startswith('[INFO]')]
    litmus_results = []
    for output in outputs:
        name = re.findall(re.compile('Test (.*?) .*'), output)[0]
        # print('_____________________')
        # print('output')
        # print(output)
        # print('_____________________')
        states, pos_cnt, neg_cnt, time_cost = parse_one_herd_output(output)
        states = [LitmusState(s) for s in states]
        states.sort(key=lambda s: str(s))
        litmus_result = LitmusResult(name, states=states, pos_cnt=pos_cnt, neg_cnt=neg_cnt, time_cost=time_cost)
        litmus_results.append(litmus_result)
    return litmus_results


def parse_chip_log(filepath) -> List[LitmusResult]:
    """
    Parse a log file generated from a real chip.
    :param filepath: the path of log file
    :return: a list of LitmusResult
    """
    from src.litmusbayes.litmus import LitmusState

    log_content = file_util.read_file(filepath)
    litmus_strs = regex_util.findall(r"% Results for .*? %\n.*?Time .*?\n", log_content, re.S)
    if len(litmus_strs) == 0:
        return []
    assert len(litmus_strs) > 0, f'[ERROR] find no litmus test execution logs in {filepath}.'
    litmus_results: List[LitmusResult] = []  # [LitmusChipResult(s) for s in litmus_strs]
    for litmus_str in litmus_strs:
        litmus_path, litmus_name, litmus_code = parse_basic_info_from_chip_output(litmus_str)
        name = re.findall(re.compile('RISCV (.*?)\n.*'), litmus_str)[0]
        numbers, states, pos_cnt, neg_cnt, time_cost = parse_state_from_chip_output(litmus_str)
        states = list(zip(numbers, states))
        states = [LitmusState(s, num) for num,s in states]
        states.sort(key=lambda s: str(s))
        litmus_result = LitmusResult(name, states, pos_cnt, neg_cnt, time_cost, litmus_code=litmus_code)
        if litmus_result in litmus_results:
            # print(f"[INFO] merge new result for {name}")
            litmus_results[litmus_results.index(litmus_result)].union(litmus_result)  # merge new result
        else:
            litmus_results.append(litmus_result)
    return litmus_results

def parse_chip_log_not_trans(filepath) -> List[LitmusResult]:
    """
    Parse a log file generated from a real chip.
    :param filepath: the path of log file
    :return: a list of LitmusResult
    """
    from src.litmusbayes.litmus import LitmusState

    log_content = file_util.read_file(filepath)
    litmus_strs = regex_util.findall(r"% Results for .*? %\n.*?Time .*?\n", log_content, re.S)
    assert len(litmus_strs) > 0, f'[ERROR] find no litmus test execution logs in {filepath}.'
    litmus_results: List[LitmusResult] = []  # [LitmusChipResult(s) for s in litmus_strs]

    litmus_code_len_dict = {}
    for litmus_str in litmus_strs:
        litmus_path, litmus_name, litmus_code = parse_basic_info_from_chip_output(litmus_str)
        if litmus_name in litmus_code_len_dict:
            litmus_code_len_dict[litmus_name] = min(len(litmus_code), litmus_code_len_dict[litmus_name])
        else:
            litmus_code_len_dict[litmus_name] = len(litmus_code)
    for litmus_str in litmus_strs:
        litmus_path, litmus_name, litmus_code = parse_basic_info_from_chip_output(litmus_str)
        if len(litmus_code) > litmus_code_len_dict[litmus_name]:
            continue
        name = re.findall(re.compile('RISCV (.*?)\n.*'), litmus_str)[0]
        numbers, states, pos_cnt, neg_cnt, time_cost = parse_state_from_chip_output(litmus_str)
        if sum(numbers) < 100000:
            continue
        states = list(zip(numbers, states))
        states = [LitmusState(s, num) for num,s in states]
        states.sort(key=lambda s: str(s))
        litmus_result = LitmusResult(name, states, pos_cnt, neg_cnt, time_cost, litmus_code=litmus_code)
        if litmus_result in litmus_results:
            # print(f"[INFO] merge new result for {name}")
            litmus_results[litmus_results.index(litmus_result)].union(litmus_result)  # merge new result
        else:
            litmus_results.append(litmus_result)
    return litmus_results

def parse_basic_info_from_chip_output(litmus_str):
    """
    litmus_str example:
    % Results for tests/non-mixed-size/HAND/2+2Swap.litmus %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RISCV 2+2Swap
    "PodWW Wse PodWW Wse"
    {0:x6=x; 0:x8=y; 1:x6=y; 1:x8=x;}
    P0                    | P1                    ;
    ori x5,x0,2           | ori x5,x0,2           ;
    amoswap.w x10,x5,(x6) | amoswap.w x10,x5,(x6) ;
    ori x7,x0,1           | ori x7,x0,1           ;
    amoswap.w x11,x7,(x8) | amoswap.w x11,x7,(x8) ;

    exists (x=2 /\ y=2 /\ 0:x10=1 /\ 0:x11=0 /\ 1:x10=1 /\ 1:x11=0)
    Generated assembler
    """
    matches = re.findall(re.compile(r"% Results for (.*?) %\n.*?RISCV (.*?)\n", re.S), litmus_str)
    assert len(matches) == 1, f'[ERROR] incorrect matches in {litmus_str}'
    result = matches[0]
    litmus_path = result[0]
    litmus_name = result[1]
    litmus_code_results = re.findall(re.compile(r"(RISCV .*?\n.*?\n)Generated assembler", re.S), litmus_str)
    if len(litmus_code_results) == 0:
        raise Exception(f"Error in parsing litmus str: {litmus_str}. May encounter incompatible encoding.")
    litmus_code = litmus_code_results[0].strip()
    return litmus_path, litmus_name, litmus_code


def parse_state_from_chip_output(litmus_str):
    """
    litmus_str example:
    Test 2+2Swap Allowed
    Histogram (3 states)
    598334:>0:x10=0; 0:x11=2; 1:x10=0; 1:x11=2; x=1; y=1;
    255189:>0:x10=1; 0:x11=2; 1:x10=0; 1:x11=0; x=2; y=1;
    146477:>0:x10=0; 0:x11=0; 1:x10=1; 1:x11=2; x=1; y=2;
    No

    Witnesses
    Positive: 0, Negative: 1000000
    Condition exists (x=2 /\ y=2 /\ 0:x10=1 /\ 0:x11=0 /\ 1:x10=1 /\ 1:x11=0) is NOT validated
    Hash=760e481b3c7b9ad1c78990727e5fcf50
    Observation 2+2Swap Never 0 1000000
    Time 2+2Swap 113.00
    """
    matches = re.findall(
        re.compile(
            r"\n.*?Histogram \((.*?) states\)\n.*?Positive: (.*?), Negative: (.*?)\n.*?\nTime .*? (.*?)\n",
            re.S), litmus_str)
    assert len(matches) == 1, f'[ERROR] incorrect matches in {litmus_str}'
    result = matches[0]
    states_cnt = int(result[0])
    pos_cnt = int(result[1])
    neg_cnt = int(result[2])
    time_cost = float(result[3])

    # output = re.findall(re.compile(r"(\nHistogram .*)", re.S), litmus_str)[0].strip()
    # parse states list. e.g., 16087 :>0:x7=3; 0:x8=0; 1:x7=0; 1:x8=0; x=2;
    """
    Histogram (2 states)
    97    :>0:x28=1; x=1;
    999903*>0:x28=0; x=2;
    Ok
    """
    states_str = re.findall(re.compile(r"(\nHistogram .*?)\nWitnesses\n", re.S), litmus_str)[0]
    states_str_list = re.findall(re.compile(".*?>(.*?)\n"), states_str)
    numbers = re.findall(r'(\d+)\s*[:*]?>', states_str)
    numbers = [int(number) for number in numbers]
    assert len(states_str_list) == len(numbers), 'the len must equal '
    state_dict_list = get_state_dict_list_from_str(states_str_list)
    return numbers, state_dict_list, pos_cnt, neg_cnt, time_cost
