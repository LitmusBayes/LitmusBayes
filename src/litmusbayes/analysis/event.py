
"""The Memory Access Events"""

from src.litmusbayes.prog.inst import *
from src.litmusbayes.prog.types import EType


class Event:
    """The memory access event.

    Parameters
    ------------
    inst: SSAInst
        The corresponding SSAInst.
    etype: EType
        The event type.
    addr: A z3 expression
        The memory address.
    data: A z3 expression
        The data read from or write to the memory.
    """

    def __init__(self,
                 inst: SSAInst = None,
                 etype: EType = EType.Unknown,
                 addr=None, data=None, pid=None):
        self.inst, self.type = inst, etype
        self.addr, self.value = addr, data
        self.pid = pid

    def is_amo(self):
        return isinstance(self.inst.inst, AmoInst) and \
            self.inst.name.startswith('amo')

    @property
    def idx(self):
        return self.inst.idx if self.inst else -1

    @property
    def signature(self):
        pid = '--' if self.pid < 0 else 'P' + str(self.pid)
        return f"{pid}-{self.idx}"

    @property
    def width(self):
        inst = self.inst.inst
        assert isinstance(inst, MemoryAccessInst)
        return inst.width

    def __repr__(self):
        # TODO: pretty print event, addr:{self.addr}\t value:{self.value}\t?
        # pid = '--' if self.pid < 0 else 'P' + str(self.pid)
        # return f"[{pid}]\t[{self.type.name}]\taddr:{self.addr}\tvalue:{self.value}\tfrom\t{self.inst}"
        return str(self)

    def __str__(self):
        pid = '--' if self.pid < 0 else 'P' + str(self.pid)
        return f'{pid}: {self.inst}'