
"""Instruction Class Definition"""
from collections import namedtuple
from typing import Optional, List

from src.litmusbayes.prog.reg import find_reg_by_name, Reg
from src.litmusbayes.prog.types import *

INST_LEN = 4

read_inst_conds = ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']
write_inst_conds = ['sb', 'sd', 'sh', 'sw']
# TODO: incomplete
read_write_inst_conds = ['amoswap', 'amoadd', 'amoor', 'amoand', 'amoxor']


# AMOMAX AMOMAXU AMOMIN AMOMINU

class Inst:
    """The instruction (in the raw format, not in ssa)."""
    global_id = 0

    def __init__(self, name: str = ''):
        self.pid = -1
        self.idx = Inst.global_id
        self.pc = self.idx * INST_LEN
        self.name = name
        self.type = IType.Unknown
        self.etype = EType.Unknown

        Inst.global_id += 1

    def init(self, inst_to_copy):
        self.idx = inst_to_copy.idx
        self.pc = inst_to_copy.pc
        # self.name = inst_to_copy.name
        self.type = inst_to_copy.type
        self.etype = inst_to_copy.etype

    @property
    def operands(self):
        raise NotImplementedError

    def get_event_type(self):
        """
        to serve Suggestion
        """
        if self.name in read_inst_conds or self.name.startswith('lr.'):
            etype = EType.Read
        elif self.name in write_inst_conds or self.name.startswith('sc.'):
            etype = EType.Write
        elif self.is_amo(self.name):
            etype = EType.ReadWrite
        else:
            etype = EType.Unknown
        return etype

    @staticmethod
    def is_amo(name):
        for cond in read_write_inst_conds:
            if name.startswith(cond):
                return True
        return False

    def get_def(self) -> Optional[Reg]:
        """Get the defined register (the destination register)."""
        raise NotImplementedError

    def get_uses(self) -> Optional[List[Reg]]:
        """Get the used registers (the source registers)."""
        raise NotImplementedError

    def get_addr(self) -> Optional[Reg]:
        """Get the address register (only for memory access instructions)."""
        return None

    def get_data(self) -> Optional[Reg]:
        """Get the data register (only for memory access instructions)."""
        return None

    def mnemonic(self, operands) -> str:
        """Get the mnemonic of the instruction (e.g., only for memory access instructions)."""
        raise NotImplementedError

    def __repr__(self):
        return f'<0x{"{:0>2X}".format(self.pc)}> {self.mnemonic(self.operands)}'
    
    def get_raw_str(self):
        return self.mnemonic(self.operands)


class RFmtInst(Inst):
    def __init__(self, name: str, rd: str, rs1: str, rs2: str):
        super().__init__(name)
        self.rd, self.rs1, self.rs2 = (find_reg_by_name(name)
                                       for name in [rd, rs1, rs2])
        self.type = IType.Normal

    @property
    def operands(self):
        return self.rd, self.rs1, self.rs2

    def get_def(self):
        return self.rd

    def get_uses(self):
        return [self.rs1, self.rs2]

    def mnemonic(self, operands):
        rd, rs1, rs2 = operands
        return f'{self.name} {rd}, {rs1}, {rs2}'


class UFmtInst(Inst):
    def __init__(self, name: str, rd: str, imm: int):
        super().__init__(name)
        self.rd, self.imm = find_reg_by_name(rd), int(imm)
        self.type = IType.Normal

    @property
    def operands(self):
        return self.rd, self.imm

    def get_def(self):
        return self.rd

    def get_uses(self):
        return []

    def mnemonic(self, operands):
        rd, imm = operands
        return f'{self.name} {rd}, {imm}'


class AmoInst(Inst):
    def __init__(self, name: str, rd: str, rs2: str, rs1: str, inst_to_copy=None):
        super().__init__(name)
        assert isinstance(rd, str) and isinstance(rs1, str) and isinstance(rs2, str)
        self.rd, self.rs2, self.rs1 = (find_reg_by_name(name)
                                       for name in [rd, rs2, rs1])
        if name.startswith('lr.'):
            self.type = IType.Lr
        elif name.startswith('sc.'):
            self.type = IType.Sc
        elif name.startswith('amo'):
            self.type = IType.Amo
        else:
            raise NotImplementedError

        if name.endswith('.aq'):
            self.flag = MoFlag.Acquire
        elif name.endswith('.aqrl') | name.endswith('.aq.rl'):
            self.flag = MoFlag.Strong
        elif name.endswith('.rl'):
            self.flag = MoFlag.Release
        else:
            self.flag = MoFlag.Relax

        if inst_to_copy is not None:
            super().init(inst_to_copy)

    @property
    def operands(self):
        return self.rd, self.rs2, self.rs1

    def get_def(self):
        return self.rd

    def get_uses(self):
        return [self.rs1, self.rs2]

    def get_data(self):
        return self.rs2

    def get_addr(self):
        return self.rs1

    def get_rs_name(self):
        return self.rs1.name

    def get_rd_name(self):
        return self.rd.name

    def get_data_src_reg_name(self):
        return self.rs2.name

    def mnemonic(self, operands):
        rd, rs2, rs1 = operands
        if self.name.startswith('lr.'):
            return f'{self.name} {rd}, 0({rs1})'
        else:
            return f'{self.name} {rd}, {rs2}, ({rs1})'


class IFmtInst(Inst):
    def __init__(self, name: str, rd: str, rs1: str, imm: int):
        super().__init__(name)
        self.rd = find_reg_by_name(rd)
        self.rs1 = find_reg_by_name(rs1)
        self.imm = imm
        self.type = IType.Normal

    @property
    def operands(self):
        return self.rd, self.rs1, self.imm

    def get_def(self):
        return self.rd

    def get_uses(self):
        return [self.rs1]

    def mnemonic(self, operands):
        rd, rs1, imm = operands
        return f'{self.name} {rd}, {rs1}, {imm}'


class BFmtInst(Inst):
    def __init__(self, name: str, rs1: str, rs2: str, imm: int = None, label: str = None):
        super().__init__(name)
        self.rs1 = find_reg_by_name(rs1)
        self.rs2 = find_reg_by_name(rs2)
        self.imm = imm
        self.label = label
        self.type = IType.Branch

    def set_label_pos(self, label_inst_idx, label_pos):
        self.label_inst_idx, self.label_pos = label_inst_idx, label_pos

    def get_label_pos(self):
        # inst_idx, after/before
        return self.label, self.label_inst_idx, self.label_pos

    @property
    def operands(self):
        return self.rs1, self.rs2, self.imm

    def get_def(self):
        return None

    def get_uses(self):
        return [self.rs1, self.rs2]

    @property
    def tgt_pc(self):
        return self.pc + self.imm

    @property
    def tgt_id(self):
        return int((self.pc + self.imm) / INST_LEN)

    def mnemonic(self, operands):
        rs1, rs2, imm = operands
        # TODO: add other pseudo branch insts
        if self.name == 'bnez':
            return f'{self.name} {rs1}, {self.label}'
        else:
            return f'{self.name} {rs1}, {rs2}, {self.label}'
        # return f'{self.name} {rs1}, {rs2}, {imm} #<{hex(self.tgt_pc)}>'


class LABEL_POS(Enum):
    # support labels index and write to suggested_program
    AFTER = 1
    BEFORE = -1


class JFmtInst(Inst):
    def __init__(self, name: str, rd: str, imm: int = None, label: str = None):
        super().__init__(name)
        self.rd = find_reg_by_name(rd)
        self.imm = imm
        self.label = label
        self.type = IType.Jump

    def set_label_pos(self, label_inst_idx, label_pos):
        self.label_inst_idx, self.label_pos = label_inst_idx, label_pos

    @property
    def operands(self):
        return self.rd, self.imm

    def get_def(self):
        return None

    def get_uses(self):
        return []

    @property
    def tgt_pc(self):
        return self.pc + self.imm

    @property
    def tgt_id(self):
        return int(self.tgt_pc / INST_LEN)

    def mnemonic(self, operands):
        rd, imm = operands
        return f'{self.name} {rd}, {imm} #<{hex(self.tgt_pc)}>'


Address = namedtuple("Address", ["base", "offset"])

_mem_width_map = {
    'lb': 1,
    'lbu': 1,
    'sb': 1,
    'lh': 2,
    'lhu': 2,
    'sh': 2,
    'lw': 4,
    'lwu': 4,
    'sw': 4,
    'ld': 8,
    'sd': 8
}


class MemoryAccessInst(Inst):
    def __init__(self, name: str, rs1: str, imm: int):
        super().__init__(name)
        self.rs1: Reg = find_reg_by_name(rs1)
        self.imm = imm

    def get_def(self):
        raise NotImplementedError

    def get_uses(self):
        raise NotImplementedError

    @property
    def addr(self):
        return Address(self.rs1, self.imm)

    def get_addr(self) -> Reg:
        return self.rs1

    @property
    def width(self):
        assert self.name in _mem_width_map, f"not implemented for {self.name}"
        return _mem_width_map[self.name]


class LoadInst(MemoryAccessInst):
    def __init__(self, name: str, rd: str, rs1: str, imm: int):
        super().__init__(name, rs1, imm)
        self.rd = find_reg_by_name(rd)
        self.type = IType.Load

    def get_rd_name(self):
        return self.rd.name

    def get_rs_name(self):
        return self.rs1.name

    @property
    def operands(self):
        return self.rd, self.rs1, self.imm

    def get_def(self):
        return self.rd

    def get_uses(self):
        return [self.rs1]

    def mnemonic(self, operands):
        rd, rs1, imm = operands
        return f'{self.name} {rd}, {imm}({rs1})'


class StoreInst(MemoryAccessInst):
    def __init__(self, name: str, rs2: str, rs1: str, imm: int):
        super().__init__(name, rs1, imm)
        self.rs2 = find_reg_by_name(rs2)
        self.type = IType.Store

    def get_rs_name(self):
        return self.rs1.name

    def get_data_src_reg_name(self):
        return self.rs2.name

    @property
    def operands(self):
        return self.rs2, self.rs1, self.imm

    def get_def(self):
        return None

    def get_uses(self):
        return [self.rs1, self.rs2]

    def get_data(self):
        return self.rs2

    def mnemonic(self, operands):
        rs2, rs1, imm = operands
        return f'{self.name} {rs2}, {imm}({rs1})'


def _fence(t: EType, flag: str):
    match t:
        case EType.Read:
            return 'r' in flag
        case EType.Write:
            return 'w' in flag
        case EType.ReadWrite:  # for mem_access inst
            return 'rw' in flag
        case _:
            return False


class MemAccessInst(Inst):
    def __init__(self, mem_access_op):
        super().__init__('mem')
        self.name = 'mem'
        self.mem_access_op = mem_access_op

        if mem_access_op == 'w':
            self.etype = EType.Write
        elif mem_access_op == 'r':
            self.etype = EType.Read
        elif mem_access_op == 'rw':
            self.etype = EType.ReadWrite
        else:
            raise Exception("unknown mem_access_op type")

    @property
    def operands(self):
        return []

    def get_def(self):
        return None

    def get_uses(self):
        return []

    def mnemonic(self, operands):
        return f'{self.name} {self.mem_access_op}'


class FenceInst(Inst):
    def __init__(self, name: str, pre: str = 'rw', suc: str = 'rw'):
        super().__init__(name)
        self.pre, self.suc = pre, suc
        self.type = IType.Fence

    def ordered_pre(self, t: EType) -> bool:
        return _fence(t, self.pre)

    def ordered_suc(self, t: EType) -> bool:
        return _fence(t, self.suc)

    def ordered(self, t1: EType, t2: EType) -> bool:
        return self.ordered_pre(t1) and self.ordered_suc(t2)

    @property
    def operands(self):
        return []

    def get_def(self):
        return None

    def get_uses(self):
        return []

    def mnemonic(self, operands):
        return f'{self.name} {self.pre}, {self.suc}'


class FenceTsoInst(Inst):
    def __init__(self):
        super().__init__('fence.tso')
        self.type = IType.FenceTso

    def ordered(self, t1: EType, t2: EType) -> bool:
        # w, w   r, r/w
        return not (t1 == EType.Write and t2 == EType.Read)

    @property
    def operands(self):
        return []

    def get_def(self):
        return None

    def get_uses(self):
        return []

    def mnemonic(self, operands):
        return f'{self.name}'


class FenceIInst(Inst):
    def __init__(self):
        super().__init__('fence.i')
        self.type = IType.FenceI

    def ordered(self, t1: EType, t2: EType) -> bool:
        return False

    @property
    def operands(self):
        return []

    def get_def(self):
        return None

    def get_uses(self):
        return []

    def mnemonic(self, operands):
        return f'{self.name}'


class JalrInst(Inst):
    def __init__(self, name: str, rd: str, rs1: str, imm: int):
        super().__init__(name)
        self.rd = find_reg_by_name(rd)
        self.rs1 = find_reg_by_name(rs1)
        self.imm = imm
        self.type = IType.Jump

    @property
    def operands(self):
        return self.rd, self.rs1, self.imm

    def get_def(self):
        return self.rd

    def get_uses(self):
        return [self.rs1]

    @property
    def tgt_id(self):
        return None

    @property
    def tgt_pc(self):
        return None

    def mnemonic(self, operands):
        rd, rs1, imm = operands
        return f'{self.name} {rd}, {imm}({rs1})'


class Label:
    def __init__(self, label, idx):
        self.label = label
        self.idx = idx

    def __str__(self):
        return f'{self.label}:'


class SSAInst:
    """
    build from an Inst
    """

    def __init__(self, inst: Inst, idx: int = -1, rmap=None):
        self.inst = inst
        self.branch_taken = None
        self.sc_succeed = None # for sc
        self.rmap = {} if rmap is None else rmap  # reg -> BitVec
        self.idx = idx

    @property
    def type(self):
        return self.inst.type

    @property
    def pc(self):
        return self.inst.pc

    @property
    def name(self):
        return self.inst.name if self.inst else 'none'

    def get_def(self):
        if self.sc_succeed is False:
            # there is no dst register if sc fails
            return None
        d = self.inst.get_def()
        return self.rmap[d] if d is not None else None

    def get_uses(self):
        return [self.rmap[u] for u in self.inst.get_uses()]

    def get_data(self):
        d = self.inst.get_data()
        return self.rmap[d] if d is not None else None

    def get_addr(self):
        a = self.inst.get_addr()
        return self.rmap[a] if a is not None else None

    def verify(self):
        if isinstance(self.inst, BFmtInst):
            assert self.branch_taken is not None, f'need branch_taken flag for {self}'

    @property
    def operands(self):
        return [self.rmap[op] if isinstance(op, Reg) and op in self.rmap else op
                for op in self.inst.operands]

    def __repr__(self):
        return f'<0x{"{:0>2X}".format(self.inst.pc)}>\t{self.inst.mnemonic(self.operands)}\t'
