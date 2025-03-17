from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.utils.cmd_util import run_cmd_and_log

#[mem, barrier, alloc, detached, thread, launch, affinity, stride, contiguous, noalign, perple]
# 这里不考虑stride和perple，需要依次修改其余所有的模式
# mem_list = ["direct", "indirect"]
# barrier_list = ["user", "userfence", "user2", "userfence2", "pthread", "none", "timebase"]
# smtmode_list = ["none", "seq", "end"] # default none
# alloc_list = ["dynamic", "before"] # default dynamic, static some time error
# speedcheck_list = ["no", "some", "all"] # default no
# thread_list = ["std", "detached", "cached"] # default std
# launch_list = ["changing", "fixed"] # default changing
# affinity_list = ["none", "random", "incr1", "incr2", "incr3"]
# noalign_list = ["none", "all"] # default changing

# diff -r /home/whq/Desktop/code_list/perple_test/stride/LB /home/whq/Desktop/code_list/perple_test/stride/LBbarrier_1 > diff/barrier1.txt
if __name__ == '__main__':
    litmus_name = "LB"
    dict = {"noalign":1,"contiguous":1}
    suffix = ""
    for k,v in dict.items():
        suffix += f"{k}_{v}"
    # suffix = "barrier_1"
    litmus_path = f"/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive/{litmus_name}.litmus"
    litmus_dir = f'/home/whq/Desktop/code_list/perple_test/stride/{litmus_name}{suffix}'
    params = LitmusParams()
    params.apply_standard_form()
    params.set_riscv_gcc()
    params.append_by_dict(dict)
    print(params.to_dict())
    print(params.to_litmus7_format(litmus_path, litmus_dir))
    run_cmd_and_log(params.to_litmus7_format(litmus_path, litmus_dir))
    run_cmd_and_log(f"diff -r /home/whq/Desktop/code_list/perple_test/stride/{litmus_name} /home/whq/Desktop/code_list/perple_test/stride/{litmus_name}{suffix} > diff/{suffix}.txt")