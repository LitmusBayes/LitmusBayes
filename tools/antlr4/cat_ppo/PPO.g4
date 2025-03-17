/*
    As compared to PPOV1, this version represent ppo2 in a more natural way

    support ppo2: ([R];po-loc\(po-loc?;[W];po-loc);[R])\rsw
    ([R];po-loc\(po-loc;[W];po-loc);[R])\rsw

(* Overlapping-Address Orderings *)
let r1 = [M];po-loc;[W]
and r2 = ([R];po-loc-no-w;[R]) \ rsw     -> ([R];po-loc\(po-loc;[W];po-loc);[R])\rsw
and r3 = [AMO|X];rfi;[R]
(* Explicit Synchronization *)
and r4 = fence
and r5 = [AQ];po;[M]
and r6 = [M];po;[RL]
and r7 = [RCsc];po;[RCsc]
and r8 = rmw
(* Syntactic Dependencies *)
and r9 = [M];addr;[M]
and r10 = [M];data;[W]
and r11 = [M];ctrl;[W]
(* Pipeline Dependencies *)
and r12 = [M];(addr|data);[W];rfi;[R]
and r13 = [M];addr;[M];po;[W]
*/

/*
2024-1-29 17:01:26

let r1 = M;po&loc;W
and r2 = (R;po-loc-no-w;R) \ rsw     -> (R;po&loc\(po&loc;W;po&loc);R)\rsw
and r3 = AMO|X;rfi;R
(* Explicit Synchronization *)
and r4 = fence
and r5 = AQ;po;M
and r6 = M;po;RL
and r7 = RCsc;po;RCsc
and r8 = rmw
(* Syntactic Dependencies *)
and r9 = M;addr;M
and r10 = M;data;W
and r11 = M;ctrl;W
(* Pipeline Dependencies *)
and r12 = M;(addr|data);W;rfi;R
and r13 = M;addr;M;po;W

Some notes:2024-1-15
    {(e1,e1)} ; {(e1,e2)  ......}  -> {(e1,e2)} Concatenation
    {(e1,e1)} & {(e1,e2)  ......}  -> {} intersection
    po-loc = po & loc
    {(e1,e2)} ; {(e2,e3)  ......} = (e1,e3)

    W&addr is None
    W(e1,e1);rmw(e1,e2) -> e1,e2
    addr(e1, e2)&rmw(e1, e2) -> e1,e2

2024-1-30 17:50:39
types: mem_types(e1,e1); relations(e1,e2)

operands must have same types:
|
&
\

operands must have different types:
;
*/

grammar PPO;
options { language=Python3; }

//s : expr EOF ;

expr: '(' expr ')'
    | expr '|' expr
    | expr '&' expr
    | expr '\\' expr
    | expr ';' expr
    | RELATION
    ;

// we do not need (expr) at present, as () is used to identify operation order but the AST tree stucture already specify the operation order.
//expr
//    : expr '|' expr
//    | expr '&' expr
//    | expr '\\' expr
//    | expr ';' expr
//    | RELATION
//    ;

RELATION
    : 'M'
    | 'RCsc'
    | 'AQ'
    | 'RL'
    | 'AQRL'
    | 'R'
    | 'W'
    | 'AMO'
    | 'X'
    | 'XSc'
    | 'XLr'
    | 'rmw'
    | 'addr'
    | 'data'
    | 'ctrl'
    | 'po'
    | 'loc'
    | 'rf'
    | 'rfi'
    | 'rfe'
    | 'rsw'
    | 'co'
    | 'coi'
    | 'coe'
    | 'fr'
    | 'fri'
    | 'fre'
    | 'po-loc'
    | 'fence_rw_rw'
    | 'fence_rw_w'
    | 'fence_rw_r'
    | 'fence_r_rw'
    | 'fence_r_r'
    | 'fence_r_w'
    | 'fence_w_rw'
    | 'fence_w_w'
    | 'fence_w_r'
    | 'fence_tso'
    | 'fence'
    ;

//    | 'fr';
//   TODO: need fr or not?


// Whitespace and comments
// refer to: https://github.com/antlr/grammars-v4/blob/master/java/java20/Java20Lexer.g4
WS: [ \t\r\n\u000C]+ -> skip;
COMMENT: '/*' .*? '*/' -> channel(HIDDEN);
LINE_COMMENT: '//' ~[\r\n]* -> channel(HIDDEN);