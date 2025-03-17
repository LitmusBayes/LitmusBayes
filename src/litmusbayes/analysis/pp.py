
"""Parallel Path"""


class ParallelPath:
    def __init__(self, paths, init_conds=None):
        """
        :param paths: [path_p0, path_p1, ...]
        :param init_conds: initial conditions generated from init part of the litmus test
        """
        from src.litmusbayes.analysis import SymbolicExecutor, path_to_ssa
        self.paths = paths
        self.se = SymbolicExecutor()
        # print('init_conds',init_conds)
        if init_conds:
            for cond in init_conds:
                self.se.solver.add(cond)
        # print('init conds',self.se.solver.assertions())
        for pid, path in enumerate(self.paths):
            ssas = path_to_ssa(path, pid)
            self.se.run(ssas, pid)
            # print('pp model',self.se.solver.assertions())

    def __repr__(self):
        return '\n'.join([f'P{i}: {p}' for i, p in enumerate(self.paths)])
