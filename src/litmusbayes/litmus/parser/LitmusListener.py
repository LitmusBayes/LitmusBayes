# Generated from Litmus.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .LitmusParser import LitmusParser
else:
    from LitmusParser import LitmusParser

# This class defines a complete listener for a parse tree produced by LitmusParser.
class LitmusListener(ParseTreeListener):

    # Enter a parse tree produced by LitmusParser#entry.
    def enterEntry(self, ctx:LitmusParser.EntryContext):
        pass

    # Exit a parse tree produced by LitmusParser#entry.
    def exitEntry(self, ctx:LitmusParser.EntryContext):
        pass


    # Enter a parse tree produced by LitmusParser#init.
    def enterInit(self, ctx:LitmusParser.InitContext):
        pass

    # Exit a parse tree produced by LitmusParser#init.
    def exitInit(self, ctx:LitmusParser.InitContext):
        pass


    # Enter a parse tree produced by LitmusParser#var_decl.
    def enterVar_decl(self, ctx:LitmusParser.Var_declContext):
        pass

    # Exit a parse tree produced by LitmusParser#var_decl.
    def exitVar_decl(self, ctx:LitmusParser.Var_declContext):
        pass


    # Enter a parse tree produced by LitmusParser#reg_decl.
    def enterReg_decl(self, ctx:LitmusParser.Reg_declContext):
        pass

    # Exit a parse tree produced by LitmusParser#reg_decl.
    def exitReg_decl(self, ctx:LitmusParser.Reg_declContext):
        pass


    # Enter a parse tree produced by LitmusParser#final.
    def enterFinal(self, ctx:LitmusParser.FinalContext):
        pass

    # Exit a parse tree produced by LitmusParser#final.
    def exitFinal(self, ctx:LitmusParser.FinalContext):
        pass


    # Enter a parse tree produced by LitmusParser#location.
    def enterLocation(self, ctx:LitmusParser.LocationContext):
        pass

    # Exit a parse tree produced by LitmusParser#location.
    def exitLocation(self, ctx:LitmusParser.LocationContext):
        pass


    # Enter a parse tree produced by LitmusParser#filter.
    def enterFilter(self, ctx:LitmusParser.FilterContext):
        pass

    # Exit a parse tree produced by LitmusParser#filter.
    def exitFilter(self, ctx:LitmusParser.FilterContext):
        pass


    # Enter a parse tree produced by LitmusParser#observed_var.
    def enterObserved_var(self, ctx:LitmusParser.Observed_varContext):
        pass

    # Exit a parse tree produced by LitmusParser#observed_var.
    def exitObserved_var(self, ctx:LitmusParser.Observed_varContext):
        pass


    # Enter a parse tree produced by LitmusParser#cond_expr.
    def enterCond_expr(self, ctx:LitmusParser.Cond_exprContext):
        pass

    # Exit a parse tree produced by LitmusParser#cond_expr.
    def exitCond_expr(self, ctx:LitmusParser.Cond_exprContext):
        pass


    # Enter a parse tree produced by LitmusParser#cond_term.
    def enterCond_term(self, ctx:LitmusParser.Cond_termContext):
        pass

    # Exit a parse tree produced by LitmusParser#cond_term.
    def exitCond_term(self, ctx:LitmusParser.Cond_termContext):
        pass


    # Enter a parse tree produced by LitmusParser#cond.
    def enterCond(self, ctx:LitmusParser.CondContext):
        pass

    # Exit a parse tree produced by LitmusParser#cond.
    def exitCond(self, ctx:LitmusParser.CondContext):
        pass


    # Enter a parse tree produced by LitmusParser#reg_cond.
    def enterReg_cond(self, ctx:LitmusParser.Reg_condContext):
        pass

    # Exit a parse tree produced by LitmusParser#reg_cond.
    def exitReg_cond(self, ctx:LitmusParser.Reg_condContext):
        pass


    # Enter a parse tree produced by LitmusParser#var_cond.
    def enterVar_cond(self, ctx:LitmusParser.Var_condContext):
        pass

    # Exit a parse tree produced by LitmusParser#var_cond.
    def exitVar_cond(self, ctx:LitmusParser.Var_condContext):
        pass


    # Enter a parse tree produced by LitmusParser#addr_cond.
    def enterAddr_cond(self, ctx:LitmusParser.Addr_condContext):
        pass

    # Exit a parse tree produced by LitmusParser#addr_cond.
    def exitAddr_cond(self, ctx:LitmusParser.Addr_condContext):
        pass



del LitmusParser