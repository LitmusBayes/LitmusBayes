import json


class MemoryStress:
    """Configuration class for memory stress testing."""

    def __init__(self, config=None):
        # Flag indicating whether memory stress is enabled
        self.start_stress = config is not None

        # Ensure config is a dictionary
        config = config or {}

        # Whether to run a pretest stress phase
        self.pretest_stress = config.get("pretest_stress", False)

        # Target number of stress iterations
        self.target_number = config.get("target_number", 1)

        # Access pattern: 0 = LL, 1 = LW, 2 = WL, 3 = WW
        self.access_pattern = config.get("access_pattern", 0)

        # Number of stress lines used in the test
        self.stress_lines = config.get("stress_lines", 512)

    def to_dict(self):
        return self.__dict__

mem_list = ["direct", "indirect"]
barrier_list = ["user", "userfence", "user2", "userfence2", "pthread", "none", "timebase"]
smtmode_list = ["none", "seq", "end"] # default none
alloc_list = ["dynamic", "before"] # default dynamic, static some time error
speedcheck_list = ["no", "some", "all"] # default no
thread_list = ["std", "detached", "cached"] # default std
launch_list = ["changing", "fixed"] # default changing
affinity_list = ["none", "random", "incr1", "incr2", "incr3"]
noalign_list = ["none", "all"] # default changing
stride_list = [1, 31, 133]
def to_dict_by_vector(vec):
    # [mem, barrier, alloc, detached, thread, launch, affinity, stride, contiguous, noalign, perple]
    # [0,    2,       0,        0,       0,      0,     2,       0,       0,           0,     0]
    attr_dict = {}
    attr_dict["mem"] = mem_list[vec[0]]
    attr_dict["barrier"] = barrier_list[vec[1]]
    attr_dict["alloc"] = alloc_list[vec[2]]
    attr_dict["detached"] = (True if vec[3] == 1 else False)
    attr_dict["thread"] = thread_list[vec[4]]
    attr_dict["launch"] = launch_list[vec[5]]
    attr_dict["affinity"] = affinity_list[vec[6]]
    attr_dict["stride"] = stride_list[vec[7]]
    attr_dict["contiguous"] = (True if vec[8] == 1 else False)
    attr_dict["noalign"] = noalign_list[vec[9]]
    attr_dict["perple"] = (True if vec[10] == 1 else False)
    return attr_dict


class LitmusParams:
    """Top-level parameters for a litmus test run."""

    def __init__(self, config=None):
        config = config or {}

        # dir_name 一般是标识符，仍然需要构造
        self.dir_name = str(config)

        # 遍历 config，动态设置属性
        for key, value in config.items():
            setattr(self, key, value)

        # 特殊字段，如果用户提供 memory_stress=None，就需要替换成 MemoryStress()
        # 否则不自动创建
        if "memory_stress" in config and config["memory_stress"] is None:
            self.memory_stress = MemoryStress()

    def to_dict(self):
        result = {}

        for k, v in self.__dict__.items():
            # 跳过 dir_name，因为这个是内部字段
            if k == "dir_name":
                continue

            # 带 to_dict 方法的对象（如 MemoryStress）
            if hasattr(v, "to_dict") and callable(v.to_dict):
                result[k] = v.to_dict()
            else:
                result[k] = v

        return result

    def apply_standard_form(self):
        self.affinity = getattr(self, "affinity", 2)
        self.force_affinity = getattr(self, "force_affinity", True)
        self.gcc = getattr(self, "gcc", "riscv64-unknown-linux-gnu-gcc")
        self.mem = getattr(self, "mem", 0)  # direct
        self.barrier = getattr(self, "barrier", 1)  # userfence
        self.stride = getattr(self, "stride", 1)
        self.s = getattr(self, "s", 100)
        self.r = getattr(self, "r", 10)
        self.smtmode = getattr(self, "smtmode", 1)  # seq
        self.smt = getattr(self, "smt", 2)
        self.a = getattr(self, "a", 4)  # avail

    def set_riscv_gcc(self):
        self.gcc = getattr(self, "gcc", "riscv64-unknown-linux-gnu-gcc")
        self.a = getattr(self, "a", 4)  # avail
        self.s = getattr(self, "s", 100)
        self.r = getattr(self, "r", 10)
        self.force_affinity = getattr(self, "force_affinity", True)

    def to_litmus7_format(self, litmus_file_path, out_dir):

        # 仅当属性存在时才拼接参数
        def opt(flag, attr_name, transform=lambda x: x):
            if hasattr(self, attr_name):
                return f"{flag} {transform(getattr(self, attr_name))}"
            return ""

        cmd = [
            f"mkdir -p {out_dir}; ",
            "eval $(opam env);",
            "litmus7",
            "-carch RISCV",
            "-limit true",
            opt("-affinity", "affinity", lambda x: affinity_list[x]),
            opt("-force_affinity", "force_affinity", lambda x: 'true' if x else 'false'),
            opt("-mem", "mem", lambda x: mem_list[x]),
            opt("-p", "p"),
            opt("-barrier", "barrier", lambda x: barrier_list[x]),
            opt("-detached", "detached", lambda x: 'true' if x else 'false'),
            opt("-thread", "thread", lambda x: thread_list[x]),
            opt("-launch", "launch", lambda x: launch_list[x]),
            opt("-stride", "stride", lambda x: stride_list[x]),
            opt("-size_of_test", "s"),
            opt("-number_of_run", "r"),
            "-driver C",
            opt("-gcc", "gcc"), # riscv64-unknown-linux-gnu-gcc
            "-ccopts -O2",
            "-linkopt -static",
            opt("-smtmode", "smtmode", lambda x: smtmode_list[x]),
            opt("-smt", "smt"),
            opt("-avail", "a"),
            opt("-alloc", "alloc", lambda x: alloc_list[x]),
            opt("-doublealloc", "doublealloc", lambda x: 'true' if x else 'false'),
            opt("-contiguous", "contiguous", lambda x: 'true' if x else 'false'),
            opt("-noalign", "noalign", lambda x: noalign_list[x]),
            opt("-speedcheck", "speedcheck", lambda x: speedcheck_list[x]),
            opt("-ascall", "ascall", lambda x: 'true' if x else 'false'),
            opt("-loop", "loop"),
            litmus_file_path,
            f"-o {out_dir}",
        ]

        # 去掉空参数并 join
        return " ".join(arg for arg in cmd if arg.strip())

    def to_vector(self):
        """
        Output vector in the order:
        [mem, barrier, alloc, detached, thread, launch, affinity, stride, contiguous, noalign, perple]
        Missing attributes default to 0.
        """

        def get_int(attr):
            if hasattr(self, attr):
                v = getattr(self, attr)
                # Boolean → int
                if isinstance(v, bool):
                    return 1 if v else 0
                return int(v)
            return 0
        return [
            get_int("mem"),
            get_int("barrier"),
            get_int("alloc"),
            get_int("detached"),
            get_int("thread"),
            get_int("launch"),
            get_int("affinity"),
            get_int("stride"),
            get_int("contiguous"),
            get_int("noalign"),
            get_int("perple")
        ]

    def from_vector(self, vec):
        """
        Load vector values back into fields in order:
        [mem, barrier, alloc, detached, thread, launch, affinity, stride, contiguous, noalign, perple]
        """
        fields = [
            "mem",
            "barrier",
            "alloc",
            "detached",
            "thread",
            "launch",
            "affinity",
            "stride",
            "contiguous",
            "noalign",
            "perple"
        ]

        for name, value in zip(fields, vec):
            # detached 本质是布尔值
            if value == -1:
                continue
            if name == "detached" or name == "perple":
                setattr(self, name, bool(value))
            else:
                setattr(self, name, int(value))

        return self
    def append_by_dict(self, config):
        for key, value in config.items():
            setattr(self, key, value)

    def is_perple(self):
        if hasattr(self, "perple") and getattr(self, "perple") == True:
            return True
        return False

    def __str__(self):
        return "_".join(str(x) for x in self.to_vector())


def dict_to_json(data):
    return json.dumps(data, indent=4)

if __name__ == "__main__":
    params = LitmusParams()
    params.apply_standard_form()
    print(dict_to_json(params.to_dict()))
    print(params.to_litmus7_format("1","2"))

    #
    print("test change")
    params = LitmusParams({"mem": 1})
    params.apply_standard_form()
    print(dict_to_json(params.to_dict()))
    print(params.to_litmus7_format("1","2"))

    print("test to vector")
    params = LitmusParams()
    params.apply_standard_form()
    print(dict_to_json(params.to_dict()))
    print(params.to_litmus7_format("1","2"))
    print(params.to_vector())

    print("test to vector")
    params = LitmusParams({"mem": 1, "barrier": 3})
    params.apply_standard_form()
    print(dict_to_json(params.to_dict()))
    print(params.to_litmus7_format("1","2"))
    print(params.to_vector())

    print("test from vector")
    params = LitmusParams()
    params.apply_standard_form()
    params.from_vector([1, 3, 0, 0, 0, 0, 2, 1, 0, 0])
    print(dict_to_json(params.to_dict()))
    print(params.to_litmus7_format("1","2"))
    print(params.to_vector())