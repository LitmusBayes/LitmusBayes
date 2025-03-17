import itertools
import random

from src.litmusbayes.bayes.litmus_params import LitmusParams


class LitmusParamSpace:
    """
    管理 litmus 参数空间：
    - 定义每一维的类别数量或范围
    - vector ↔ LitmusParams 映射
    """

    def __init__(self):
        # categorical lists
        self.mem_list = ["direct", "indirect"]
        self.barrier_list = ["user", "userfence", "user2", "userfence2", "pthread", "none", "timebase"]
        self.alloc_list = ["dynamic", "before"]
        self.thread_list = ["std", "detached", "cached"]
        self.launch_list = ["changing", "fixed"]
        self.affinity_list = ["none", "random", "incr1", "incr2", "incr3"]
        self.noalign_list = ["none", "all"]

        # stride 是数值空间
        self.stride_list = [1, 31, 133]
        self.dim = 11

    # --------------------------------------------------
    #   各维度取值范围（训练 RF 用）
    # --------------------------------------------------

    def get_bounds(self):
        return {
            "mem": (0, len(self.mem_list) - 1),
            "barrier": (0, len(self.barrier_list) - 1),
            "alloc": (0, len(self.alloc_list) - 1),
            "detached": (0, 1),
            "thread": (0, len(self.thread_list) - 1),
            "launch": (0, len(self.launch_list) - 1),
            "affinity": (0, len(self.affinity_list) - 1),
            "stride": (0, len(self.stride_list) - 1),
            "contiguous": (0, 1),
            "noalign": (0, len(self.noalign_list) - 1),
            "perple": (0, 0)  # to Fix
        }

    def random_vector(self, can_perple):
        """随机采样一个向量（整数编码）"""
        bounds = self.get_bounds()
        while True:
            item = [random.randint(low, high) for (low, high) in bounds.values()]
            if item[7] != 0 and item[10] == 1:
                continue
            if not can_perple and item[10] == 1:
                continue
            return item
        # return [random.randint(low, high) for (low, high) in bounds.values()]

    def get_bound_vector(self):
        vec_list = []
        vec_list.append([0,2,0,0,0,0,2,0,0,0,0])
        vec_list.append([1,1,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,0,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,3,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,4,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,5,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,6,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,1,1,0,0,0,2,0,0,0,0])
        vec_list.append([0,1,0,1,0,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,1,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,2,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,0,1,2,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,1,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,3,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,4,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,0,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,2,0,1,0,0])
        vec_list.append([0,1,0,0,0,0,2,0,0,1,0])
        vec_list.append([0,1,0,0,0,0,2,1,0,0,0])
        vec_list.append([0,1,0,0,0,0,2,2,0,0,0])
        return vec_list

    def get_vector_by_important(self):
        vec_list = []
        # vec_list.append([0,1,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,2,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,0,0,0,0,0,2,0,0,0,0])
        vec_list.append([0,3,0,0,0,0,2,0,0,0,0])

        vec_list.append([1,1,0,0,0,0,2,0,0,0,0])
        vec_list.append([1,2,0,0,0,0,2,0,0,0,0])
        vec_list.append([1,0,0,0,0,0,2,0,0,0,0])
        vec_list.append([1,3,0,0,0,0,2,0,0,0,0])

        vec_list.append([1,1,0,0,0,0,2,1,0,0,0])
        vec_list.append([1,2,0,0,0,0,2,1,0,0,0])
        vec_list.append([1,0,0,0,0,0,2,1,0,0,0])
        vec_list.append([1,3,0,0,0,0,2,1,0,0,0])

        vec_list.append([0,1,0,0,0,0,2,1,0,0,0])
        vec_list.append([0,2,0,0,0,0,2,1,0,0,0])
        vec_list.append([0,0,0,0,0,0,2,1,0,0,0])
        vec_list.append([0,3,0,0,0,0,2,1,0,0,0])

        vec_list.append([0,1,0,0,0,0,2,2,0,0,0])
        vec_list.append([0,2,0,0,0,0,2,2,0,0,0])
        vec_list.append([0,0,0,0,0,0,2,2,0,0,0])
        vec_list.append([0,3,0,0,0,0,2,2,0,0,0])

        vec_list.append([1,1,0,0,0,0,2,2,0,0,0])
        vec_list.append([1,2,0,0,0,0,2,2,0,0,0])
        vec_list.append([1,0,0,0,0,0,2,2,0,0,0])
        vec_list.append([1,3,0,0,0,0,2,2,0,0,0])

        vec_list.append([0,1,1,0,0,0,2,0,0,0,0])
        vec_list.append([0,1,0,1,0,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,1,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,2,0,2,0,0,0,0])
        vec_list.append([0,1,0,0,0,1,2,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,1,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,3,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,4,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,0,0,0,0,0])
        vec_list.append([0,1,0,0,0,0,2,0,1,0,0])
        vec_list.append([0,1,0,0,0,0,2,0,0,1,0])
        return vec_list
    # --------------------------------------------------
    # vector → parameter dict
    # --------------------------------------------------
    def vector_to_param_dict(self, vec):
        # return {
        #     "mem": self.mem_list[vec[0]],
        #     "barrier": self.barrier_list[vec[1]],
        #     "alloc": self.alloc_list[vec[2]],
        #     "detached": bool(vec[3]),
        #     "thread": self.thread_list[vec[4]],
        #     "launch": self.launch_list[vec[5]],
        #     "affinity": self.affinity_list[vec[6]],
        #     "stride": self.stride_list[vec[7]],
        #     "contiguous": bool(vec[8]),
        #     "noalign": self.noalign_list[vec[9]],
        #     "perple": bool(vec[10]),
        # }

        return {
            "mem": vec[0],
            "barrier": vec[1],
            "alloc": vec[2],
            "detached": vec[3],
            "thread": vec[4],
            "launch": vec[5],
            "affinity": vec[6],
            "stride": vec[7],
            "contiguous": vec[8],
            "noalign": vec[9],
            "perple": vec[10],
        }

    # --------------------------------------------------
    # vector → LitmusParams
    # --------------------------------------------------
    def vector_to_params(self, vec):
        return LitmusParams(config=self.vector_to_param_dict(vec))

    # --------------------------------------------------
    # 参数向量 clip 到合法区间
    # --------------------------------------------------
    def clip_vector(self, vec):
        bounds = self.get_bounds()
        clipped = []
        for (low, high), v in zip(bounds.values(), vec):
            v = max(low, min(high, int(round(v))))
            clipped.append(v)
        return clipped

    def can_pass(self, item, can_perple):

        if item[7] != 0 and item[10] == 1:
            return False
        if not can_perple and item[10] == 1:
            return False
        return True

    def get_all_combinations(self, can_perple = False):
        """
        生成参数空间的全排列（笛卡尔积）。
        返回一个列表，其中每个元素都是一个参数 list (vector)。
        """
        bounds = self.get_bounds()
        # bounds.values() 返回的是 [(0,1), (0,6), ...]
        # 我们需要把每个元组转成 range 对象： range(0, 2), range(0, 7)...
        # 注意 range 的结束位需要 +1
        ranges = [range(low, high + 1) for low, high in bounds.values()]

        # 使用 itertools.product 生成笛卡尔积
        # result 是一个迭代器，转成 list
        all_vectors = list(itertools.product(*ranges))

        # itertools.product 生成的是 tuple，为了保持和你之前 vector 一致，转成 list
        vectors = [list(vec) for vec in all_vectors]
        vecs = []
        for vec in vectors:
            if self.can_pass(vec, can_perple):
                vecs.append(vec)
        return vecs


    def get_all_combinations_for_best(self, can_perple = False):
        """
        生成参数空间的全排列（笛卡尔积）。
        返回一个列表，其中每个元素都是一个参数 list (vector)。
        """
        bounds = self.get_bounds()
        # bounds.values() 返回的是 [(0,1), (0,6), ...]
        # 我们需要把每个元组转成 range 对象： range(0, 2), range(0, 7)...
        # 注意 range 的结束位需要 +1
        ranges = [range(low, high + 1) for low, high in bounds.values()]


        # 使用 itertools.product 生成笛卡尔积
        # result 是一个迭代器，转成 list
        all_vectors = list(itertools.product(*ranges))

        # itertools.product 生成的是 tuple，为了保持和你之前 vector 一致，转成 list
        vectors = [list(vec) for vec in all_vectors]
        vecs = []
        for vec in vectors:
            if vec[1] == 4:
                continue
            if self.can_pass(vec, can_perple):
                vecs.append(vec)
        return vecs