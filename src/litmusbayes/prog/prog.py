

"""Program Class Definition"""
from copy import deepcopy
from typing import Dict

from antlr4 import *

from src.litmusbayes.prog import *
from src.litmusbayes.prog.parser.ProgramLexer import ProgramLexer
from src.litmusbayes.prog.parser.ProgramListener import ProgramListener
from src.litmusbayes.prog.parser.ProgramParser import ProgramParser


class Program:
    def __init__(self, insts=None):
        self.insts = [] if insts is None else insts
        self.labels: Dict[str, int] = {}

    def __repr__(self):
        insts_repr = [str(i) for i in self.insts]
        attach_label_before = lambda l, i: f"{l}:\n{i}"
        attach_label_after = lambda l, i: f"{i}{l}:"
        for label, label_id in self.labels.items():
            if label_id < len(insts_repr):
                attach_label = attach_label_before
            else:
                attach_label = attach_label_after
                label_id = -1
                insts_repr[-1] = attach_label_after(label, insts_repr[-1])
            insts_repr[label_id] = attach_label(label, insts_repr[label_id])
        return '\n'.join(insts_repr)
    
    def get_all_array(self):
        array = deepcopy(self.insts)
        array = [i for i in array]
        attach_label_before = {}
        attach_label_after = {}

        for label, label_id in self.labels.items():
            if label_id < len(array):
                if label_id in attach_label_before:
                    attach_label_before[label_id].append(f'{label}:')
                else:
                    attach_label_before[label_id] = [f'{label}:']
            else:
                attach_label_after.setdefault(-1, [])
                attach_label_after[-1].append(f'{label}:')

        final_array = []
        for inst in array:
            id = inst.idx
            if id in attach_label_before:
                final_array.extend(attach_label_before[id])
            final_array.append(inst)
            if id in attach_label_after:
                final_array.extend(attach_label_after[id])
        if -1 in attach_label_after:
            final_array.extend(attach_label_after[-1])
        return final_array


class ProgramParseListener(ProgramListener):
    def __init__(self):
        self.program = Program()
        Inst.global_id = 0

    def enterInst_j(self, ctx: ProgramParser.Inst_jContext):
        imm, label = ctx.IMM(), ctx.LABEL()
        inst = JFmtInst('jal', 'x0')
        if imm is not None:
            inst.imm = int(imm.getText())
        elif label is not None:
            inst.label = label.getText()
        self.program.insts.append(inst)

    def enterInst_jr(self, ctx: ProgramParser.Inst_jrContext):
        rs1 = ctx.REG().getText()
        inst = JalrInst('jalr', 'x0', rs1, 0)
        self.program.insts.append(inst)

    def enterInst_nop(self, ctx: ProgramParser.Inst_nopContext):
        inst = IFmtInst('addi', 'x0', 'x0', 0)
        self.program.insts.append(inst)

    def enterRfmt(self, ctx: ProgramParser.RfmtContext):
        rd, rs1, rs2 = (ctx.REG(i).getText() for i in range(3))
        if ctx.R_FMT_NAME():
            inst = RFmtInst(ctx.R_FMT_NAME().getText(), rd, rs1, rs2)
        else: # is an or inst
            inst = RFmtInst('or', rd, rs1, rs2)
        self.program.insts.append(inst)

    def enterAmofmt(self, ctx: ProgramParser.AmofmtContext):
        rd, rs2, rs1 = (ctx.REG(i).getText() for i in range(3))
        name = ctx.AMO_NAME().getText()
        if ctx.MO_FLAG() is not None:
            name += ctx.MO_FLAG().getText()
        inst = AmoInst(name, rd, rs2, rs1)
        self.program.insts.append(inst)

    def enterInst_fence(self, ctx: ProgramParser.Inst_fenceContext):
        pre, suc = ctx.mem_access_op(0), ctx.mem_access_op(1)
        inst = FenceInst('fence') if pre is None else FenceInst('fence', pre.getText(), suc.getText())
        self.program.insts.append(inst)

    def enterInst_fencei(self, ctx: ProgramParser.Inst_fenceiContext):
        self.program.insts.append(FenceIInst())

    def enterInst_fencetso(self, ctx: ProgramParser.Inst_fencetsoContext):
        self.program.insts.append(FenceTsoInst())

    def enterIfmt(self, ctx: ProgramParser.IfmtContext):
        rd, rs1, imm = ctx.REG(0).getText(), ctx.REG(1).getText(), int(ctx.IMM().getText())
        inst = IFmtInst(ctx.I_FMT_NAME().getText(), rd, rs1, imm)
        self.program.insts.append(inst)

    def enterMfmt(self, ctx: ProgramParser.MfmtContext):
        rd, rs1, imm = ctx.REG(0).getText(), ctx.REG(1).getText(), int(ctx.IMM().getText())
        ld_name, sd_name, jalr = ctx.LD_NAME(), ctx.SD_NAME(), ctx.JALR()
        inst = None
        if ld_name is not None:
            ld_name_str = ld_name.getText()
            if ld_name_str.startswith('lr'):
                inst = AmoInst(ld_name_str, rd, 'x0', rs1)
            else:
                inst = LoadInst(ld_name_str, rd, rs1, imm)
        elif sd_name is not None:
            inst = StoreInst(sd_name.getText(), rd, rs1, imm)
        elif jalr is not None:
            inst = JalrInst('jalr', rd, rs1, imm)
        assert inst is not None, f"[ERROR] parser: unknown MFmt instruction"
        self.program.insts.append(inst)

    def enterUfmt(self, ctx: ProgramParser.UfmtContext):
        ufmt_name, reg, imm = ctx.U_FMT_NAME().getText(), ctx.REG().getText(), int(ctx.IMM().getText())
        inst = UFmtInst(ufmt_name, reg, imm)
        assert inst is not None, f"[ERROR] parser: unknown UFmt instruction"
        self.program.insts.append(inst)

    def enterBfmt(self, ctx: ProgramParser.BfmtContext):
        rs1, rs2, imm, label = ctx.REG(0).getText(), ctx.REG(1).getText(), ctx.IMM(), ctx.LABEL()
        inst = BFmtInst(ctx.B_FMT_NAME().getText(), rs1, rs2)
        if imm is not None:
            inst.imm = int(imm.getText())
        elif label is not None:
            inst.label = label.getText()
        self.program.insts.append(inst)

    def enterBranch_pseudo_inst(self, ctx: ProgramParser.Branch_pseudo_instContext):
        # TODO: to be implemented
        name, rs1, label = ctx.BRANCH_PSEUDO_NAME().getText(), ctx.REG().getText(), ctx.LABEL().getText()
        # inst = BFmtInst(name, rs1, None, None, label)
        # self.program.insts.append(inst)

    def enterJfmt(self, ctx: ProgramParser.JfmtContext):
        rd, imm, label = ctx.REG().getText(), ctx.IMM(), ctx.LABEL()
        inst = JFmtInst('jal', rd)
        if imm is not None:
            inst.imm = int(imm.getText())
        elif label is not None:
            inst.label = label.getText()
        self.program.insts.append(inst)

    def enterLabel(self, ctx: ProgramParser.LabelContext):
        label = ctx.LABEL().getText()
        if 'locations' in label:
            return
        assert label not in self.program.labels, '[ERROR] parser: duplicated label'
        self.program.labels[label] = Inst.global_id

    def exitProg(self, ctx: ProgramParser.ProgContext):
        insts, labels = self.program.insts, self.program.labels

        # check if all labels are defined
        for inst in insts:
            if isinstance(inst, BFmtInst) or isinstance(inst, JFmtInst):
                label = inst.label
                if label is not None:
                    assert label in labels, f"[ERROR] parser: unknown label {label}"
                    inst.imm = (labels[label] - inst.idx) * 4


def parse_program(text):
    lexer = ProgramLexer(InputStream(text))
    stream = CommonTokenStream(lexer)
    parser = ProgramParser(stream)
    tree = parser.prog()
    listener = ProgramParseListener()
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    return listener.program
