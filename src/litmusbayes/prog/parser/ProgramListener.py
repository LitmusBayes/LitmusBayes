# Generated from Program.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ProgramParser import ProgramParser
else:
    from ProgramParser import ProgramParser

# This class defines a complete listener for a parse tree produced by ProgramParser.
class ProgramListener(ParseTreeListener):

    # Enter a parse tree produced by ProgramParser#prog.
    def enterProg(self, ctx:ProgramParser.ProgContext):
        pass

    # Exit a parse tree produced by ProgramParser#prog.
    def exitProg(self, ctx:ProgramParser.ProgContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst.
    def enterInst(self, ctx:ProgramParser.InstContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst.
    def exitInst(self, ctx:ProgramParser.InstContext):
        pass


    # Enter a parse tree produced by ProgramParser#label.
    def enterLabel(self, ctx:ProgramParser.LabelContext):
        pass

    # Exit a parse tree produced by ProgramParser#label.
    def exitLabel(self, ctx:ProgramParser.LabelContext):
        pass


    # Enter a parse tree produced by ProgramParser#rfmt.
    def enterRfmt(self, ctx:ProgramParser.RfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#rfmt.
    def exitRfmt(self, ctx:ProgramParser.RfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#ifmt.
    def enterIfmt(self, ctx:ProgramParser.IfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#ifmt.
    def exitIfmt(self, ctx:ProgramParser.IfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#mfmt.
    def enterMfmt(self, ctx:ProgramParser.MfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#mfmt.
    def exitMfmt(self, ctx:ProgramParser.MfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#bfmt.
    def enterBfmt(self, ctx:ProgramParser.BfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#bfmt.
    def exitBfmt(self, ctx:ProgramParser.BfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#jfmt.
    def enterJfmt(self, ctx:ProgramParser.JfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#jfmt.
    def exitJfmt(self, ctx:ProgramParser.JfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#ufmt.
    def enterUfmt(self, ctx:ProgramParser.UfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#ufmt.
    def exitUfmt(self, ctx:ProgramParser.UfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#amofmt.
    def enterAmofmt(self, ctx:ProgramParser.AmofmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#amofmt.
    def exitAmofmt(self, ctx:ProgramParser.AmofmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#pseudo.
    def enterPseudo(self, ctx:ProgramParser.PseudoContext):
        pass

    # Exit a parse tree produced by ProgramParser#pseudo.
    def exitPseudo(self, ctx:ProgramParser.PseudoContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_j.
    def enterInst_j(self, ctx:ProgramParser.Inst_jContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_j.
    def exitInst_j(self, ctx:ProgramParser.Inst_jContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_jr.
    def enterInst_jr(self, ctx:ProgramParser.Inst_jrContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_jr.
    def exitInst_jr(self, ctx:ProgramParser.Inst_jrContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_nop.
    def enterInst_nop(self, ctx:ProgramParser.Inst_nopContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_nop.
    def exitInst_nop(self, ctx:ProgramParser.Inst_nopContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fence.
    def enterInst_fence(self, ctx:ProgramParser.Inst_fenceContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fence.
    def exitInst_fence(self, ctx:ProgramParser.Inst_fenceContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fencetso.
    def enterInst_fencetso(self, ctx:ProgramParser.Inst_fencetsoContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fencetso.
    def exitInst_fencetso(self, ctx:ProgramParser.Inst_fencetsoContext):
        pass


    # Enter a parse tree produced by ProgramParser#fence_single.
    def enterFence_single(self, ctx:ProgramParser.Fence_singleContext):
        pass

    # Exit a parse tree produced by ProgramParser#fence_single.
    def exitFence_single(self, ctx:ProgramParser.Fence_singleContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fencei.
    def enterInst_fencei(self, ctx:ProgramParser.Inst_fenceiContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fencei.
    def exitInst_fencei(self, ctx:ProgramParser.Inst_fenceiContext):
        pass


    # Enter a parse tree produced by ProgramParser#branch_pseudo_inst.
    def enterBranch_pseudo_inst(self, ctx:ProgramParser.Branch_pseudo_instContext):
        pass

    # Exit a parse tree produced by ProgramParser#branch_pseudo_inst.
    def exitBranch_pseudo_inst(self, ctx:ProgramParser.Branch_pseudo_instContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_mem_access.
    def enterInst_mem_access(self, ctx:ProgramParser.Inst_mem_accessContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_mem_access.
    def exitInst_mem_access(self, ctx:ProgramParser.Inst_mem_accessContext):
        pass


    # Enter a parse tree produced by ProgramParser#mem_access_op.
    def enterMem_access_op(self, ctx:ProgramParser.Mem_access_opContext):
        pass

    # Exit a parse tree produced by ProgramParser#mem_access_op.
    def exitMem_access_op(self, ctx:ProgramParser.Mem_access_opContext):
        pass


    # Enter a parse tree produced by ProgramParser#mem_access_op_single.
    def enterMem_access_op_single(self, ctx:ProgramParser.Mem_access_op_singleContext):
        pass

    # Exit a parse tree produced by ProgramParser#mem_access_op_single.
    def exitMem_access_op_single(self, ctx:ProgramParser.Mem_access_op_singleContext):
        pass



del ProgramParser