import os

from src.litmusbayes.bayes.litmus_params import LitmusParams
from src.litmusbayes.bayes.litmus_util import test_dir
from src.litmusbayes.perple.script import build_perple_cmd, make_perple_cmd
from src.litmusbayes.utils.cmd_util import run_cmd


class LitmusCreator:

    def __init__(self, litmus_paths, output_dir):
        self.litmus_paths = litmus_paths
        self.output_dir = output_dir
        self.args = []

    def add_arg(self, arg):
        self.args.append(arg)

    def create_litmus(self):
        for litmus_path in self.litmus_paths:
            litmus_name = litmus_path.split("/")[-1].split(".")[0]
            print(litmus_name)
            for arg in self.args:
                params = LitmusParams(arg)
                arg_str = str(params)
                litmus_out_path = os.path.join(self.output_dir, f'{litmus_name}_{arg_str}')
                cmd = params.to_litmus7_format(litmus_path, litmus_out_path)
                print(cmd)
                run_cmd(cmd)
                run_cmd(f'cd {litmus_out_path};make')

litmus_dir = "/home/whq/Desktop/code_list/perple_test/delay_litmus_perple"
arg_dir = "/home/whq/Desktop/code_list/perple_test/arg_perple_delay"
if __name__ == '__main__':

    all_paths = []
    for root, dirs, files in os.walk(litmus_dir):
        for name in files:
            all_paths.append(os.path.join(root, name))

    creator = LitmusCreator(all_paths, arg_dir)
    # round 1
    # creator.add_arg({})
    # creator.add_arg({'mem':1})
    # creator.add_arg({'barrier':1})
    # creator.add_arg({'barrier':2})
    # creator.add_arg({'barrier':3})
    # creator.add_arg({'barrier':4})
    # creator.add_arg({'barrier':5})
    # creator.add_arg({'barrier':6})
    # creator.add_arg({'detached':True})
    # creator.add_arg({'thread':1})
    # creator.add_arg({'launch':1})
    # creator.add_arg({'alloc':1})
    # creator.add_arg({'alloc':2})
    # creator.add_arg({'doublealloc':True,'alloc':2})
    # creator.add_arg({'contiguous':True})
    # creator.add_arg({'noalign':"all"})
    # creator.add_arg({'speedcheck':1})
    # creator.add_arg({'speedcheck':2})
    # creator.add_arg({'ascall':True})

    # round 2
    args = [{'mem':1},{'detached':True},{'thread':1},{'launch':1},{'noalign':"all"},{'contiguous':True},{'alloc':2}]
    barrier_args = [{'barrier':5}]

    for i in range(0, len(barrier_args)):
        arg = {}
        for j in range(0, len(args)):
           creator.add_arg(args[j]| barrier_args[i])

    for i in range(0, len(barrier_args)):
        arg = {}
        for j in range(0, len(args)):
            for k  in range(0, len(args)):
                if j == k:
                    continue
                creator.add_arg(args[j]| args[k] | barrier_args[i])


    creator.create_litmus()

    for root, dirs, files in os.walk(arg_dir):
        for d in dirs:
            print(os.path.join(root, d))
            print(root)
            litmus_name = d.split("_")[0]
            print(litmus_name)
            for litmus_path in all_paths:
                if litmus_path.split("/")[-1].split(".")[0] == litmus_name:
                    make_perple_cmd(os.path.join(root, d), litmus_path)


