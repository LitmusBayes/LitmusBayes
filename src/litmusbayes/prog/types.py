
"""Basic Types"""

from enum import Enum


class IType(Enum):
    """The instruction type."""
    Normal = 0  # Normal instructions, e.g., add, mul.
    Branch = 1  # Branch instructions, e.g., beq, bne, blt.
    Jump = 2  # Jump instructions, e.g., jal, jalr.
    Load = 3  # Load instructions, e.g., ld, lw.
    Store = 4  # Store instructions, e.g., sd, sw.
    Amo = 5  # Amo instructions, e.g., amoadd.w.
    Lr = 6  # Load reserved instructions, e.g., lr.w, lr.d.
    Sc = 7  # Store conditional instructions, e.g., sc.w, sc.d.
    Fence = 8  # Fence instructions, e.g., fence rw,rw.
    FenceTso = 9  # fence.tso.
    FenceI = 10  # fence.i.
    Unknown = -1  # Unknown instruction.


class MoFlag(Enum):
    """Memory ordering flag."""
    Relax = 0
    Acquire = 1
    Release = 2
    Strong = 3


class EType(Enum):
    Unknown = -1
    Read = 1
    Write = 2
    ReadWrite = 3
    SC_Fail = 4


class QType(Enum):
    """Quantifier Type"""
    Forall = 1
    Exists = 2
