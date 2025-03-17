# Generated from PPO.g4 by ANTLR 4.12.0
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,10,28,2,0,7,0,1,0,1,0,1,0,1,0,1,0,1,0,3,0,9,8,0,1,0,1,0,1,0,
        1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,5,0,23,8,0,10,0,12,0,26,9,0,
        1,0,0,1,0,1,0,0,0,31,0,8,1,0,0,0,2,3,6,0,-1,0,3,4,5,1,0,0,4,5,3,
        0,0,0,5,6,5,2,0,0,6,9,1,0,0,0,7,9,5,7,0,0,8,2,1,0,0,0,8,7,1,0,0,
        0,9,24,1,0,0,0,10,11,10,5,0,0,11,12,5,3,0,0,12,23,3,0,0,6,13,14,
        10,4,0,0,14,15,5,4,0,0,15,23,3,0,0,5,16,17,10,3,0,0,17,18,5,5,0,
        0,18,23,3,0,0,4,19,20,10,2,0,0,20,21,5,6,0,0,21,23,3,0,0,3,22,10,
        1,0,0,0,22,13,1,0,0,0,22,16,1,0,0,0,22,19,1,0,0,0,23,26,1,0,0,0,
        24,22,1,0,0,0,24,25,1,0,0,0,25,1,1,0,0,0,26,24,1,0,0,0,3,8,22,24
    ]

class PPOParser ( Parser ):

    grammarFileName = "PPO.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "'|'", "'&'", "'\\'", "';'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "RELATION", 
                      "WS", "COMMENT", "LINE_COMMENT" ]

    RULE_expr = 0

    ruleNames =  [ "expr" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    RELATION=7
    WS=8
    COMMENT=9
    LINE_COMMENT=10

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PPOParser.ExprContext)
            else:
                return self.getTypedRuleContext(PPOParser.ExprContext,i)


        def RELATION(self):
            return self.getToken(PPOParser.RELATION, 0)

        def getRuleIndex(self):
            return PPOParser.RULE_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpr" ):
                listener.enterExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpr" ):
                listener.exitExpr(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = PPOParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 0
        self.enterRecursionRule(localctx, 0, self.RULE_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 8
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [1]:
                self.state = 3
                self.match(PPOParser.T__0)
                self.state = 4
                self.expr(0)
                self.state = 5
                self.match(PPOParser.T__1)
                pass
            elif token in [7]:
                self.state = 7
                self.match(PPOParser.RELATION)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 24
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 22
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
                    if la_ == 1:
                        localctx = PPOParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 10
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 11
                        self.match(PPOParser.T__2)
                        self.state = 12
                        self.expr(6)
                        pass

                    elif la_ == 2:
                        localctx = PPOParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 13
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 14
                        self.match(PPOParser.T__3)
                        self.state = 15
                        self.expr(5)
                        pass

                    elif la_ == 3:
                        localctx = PPOParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 16
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 17
                        self.match(PPOParser.T__4)
                        self.state = 18
                        self.expr(4)
                        pass

                    elif la_ == 4:
                        localctx = PPOParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 19
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 20
                        self.match(PPOParser.T__5)
                        self.state = 21
                        self.expr(3)
                        pass

             
                self.state = 26
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[0] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 4)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 3)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 2)
         




