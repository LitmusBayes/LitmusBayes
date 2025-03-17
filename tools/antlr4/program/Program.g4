grammar Program;
options { language=Python3; }

//prog : (label | inst)+ ;
//inst : (rfmt | ifmt | mfmt | bfmt | jfmt | ufmt | amofmt | pseudo | inst_fence | inst_fencetso | inst_fencei) ';'? ; // ('[' PROC ']')? (rfmt | ifmt | mfmt) ';'? ;
//label: LABEL ':';

// each statetement is supposed to have an ending ';'
prog : (label | inst)+ ;
inst : (rfmt | ifmt | mfmt | bfmt | jfmt | ufmt | amofmt | pseudo | inst_fence | inst_fencetso | inst_fencei | inst_mem_access) ';'? ;
label: LABEL ':' ';'?;

// TODO: here is an workaround for 'or' inst parsing (actually it is a bug of current Antlr4)
rfmt: (R_FMT_NAME REG ',' REG ',' REG) | ('or' REG ',' REG ',' REG);
ifmt: I_FMT_NAME REG ',' REG ',' IMM ;
// sw.rl /home/apr/tools/mappo/dataset/input/dataset_riscv_litmus/ATOMICS/RELAX/PodRWPX/LB+poprl+popx_has_ppo_0.litmus sw.rl
//mfmt: (LD_NAME(MO_FLAG)? | SD_NAME(MO_FLAG)? | JALR) REG ',' IMM '(' REG ')' ;
mfmt: (LD_NAME | SD_NAME | JALR) REG ',' IMM '(' REG ')' ;
bfmt: B_FMT_NAME REG ',' REG ',' (LABEL | IMM);
jfmt: 'jal' REG ',' (LABEL | IMM)  ;
ufmt: U_FMT_NAME REG ',' IMM ;
amofmt: AMO_NAME(MO_FLAG)? REG ',' REG ',' IMM?'(' REG ')';
pseudo : inst_j | inst_jr | inst_nop | branch_pseudo_inst;

inst_j: 'j' (LABEL | IMM) ;
inst_jr: 'jr' REG ;
inst_nop: 'nop' ;
inst_fence: ('fence' mem_access_op ',' mem_access_op) | fence_single;
inst_fencetso : 'fence.tso' ;
fence_single:'fence';
inst_fencei: 'fence.i' ;
branch_pseudo_inst: BRANCH_PSEUDO_NAME REG ',' LABEL;
inst_mem_access: 'mem' mem_access_op;
mem_access_op: FENCE_OP| mem_access_op_single;
mem_access_op_single: ('r' | 'w' | 'rw');
FENCE_OP: ('i' | 'o' |'io')?('r' | 'w' | 'rw');

// http://csci206sp2020.courses.bucknell.edu/files/2020/01/riscv-card.pdf
BRANCH_PSEUDO_NAME: 'bnez' | 'beqz' | 'blez' | 'bgez' | 'bltz' | 'bgtz';

JALR: 'jalr' ;

R_FMT_NAME : 'add' | 'addw' | 'and' | 'div' | 'divu' | 'divuw' | 'divw'  | 'mul' | 'mulh' | 'mulhsu' | 'mulhu' | 'mulw' | 'rem' | 'remu' | 'remuw' | 'remw'  | 'sll' | 'sllw' | 'slt' | 'sltu' | 'sra' | 'sraw' | 'srl' | 'srlw' | 'sub' | 'subw' | 'xor' ;
I_FMT_NAME : 'addi' | 'addiw' | 'andi' | 'ori' | 'slli' | 'slliw' | 'slti' | 'sltiu' | 'srai' | 'sraiw' | 'srli' | 'srliw' | 'xori' ;
B_FMT_NAME: 'beq' | 'bge' | 'bgeu' | 'blt' | 'bltu' | 'bne' ;
U_FMT_NAME: 'auipc' | 'lui' | 'li';

LD_NAME : 'lb' | 'lbu' | 'ld' | 'lh' | 'lhu' | 'lw' | 'lwu' | ('lr.w' | 'lr.d')MO_FLAG?;
SD_NAME : 'sb' | 'sd' | 'sh' | 'sw' ;
AMO_NAME : 'amoadd.d' | 'amoadd.w' | 'amoand.d' | 'amoand.w' | 'amomax.d' | 'amomax.w' | 'amomaxu.d' | 'amomaxu.w' | 'amomin.d' | 'amomin.w' | 'amominu.d' | 'amominu.w' | 'amoor.d' | 'amoor.w' | 'amoswap.d' | 'amoswap.w' | 'amoxor.d' | 'amoxor.w' | 'sc.d' | 'sc.w';
// may be aq.rl
MO_FLAG : '.aq' | '.rl' | '.aqrl' | '.aq.rl' ;

REG : 'x0' | 'x1' | 'x2' | 'x3' | 'x4' | 'x5' | 'x6' | 'x7' | 'x8' | 'x9' | 'x10' | 'x11' | 'x12' | 'x13' | 'x14' | 'x15' | 'x16' | 'x17' | 'x18' | 'x19' | 'x20' | 'x21' | 'x22' | 'x23' | 'x24' | 'x25' | 'x26' | 'x27' | 'x28' | 'x29' | 'x30' | 'x31' | 'zero' | 'ra' | 'sp' | 'gp' | 'tp' | 't0' | 't1' | 't2' | 's0' | 's1' | 'a0' | 'a1' | 'a2' | 'a3' | 'a4' | 'a5' | 'a6' | 'a7' | 's2' | 's3' | 's4' | 's5' | 's6' | 's7' | 's8' | 's9' | 's10' | 's11' | 't3' | 't4' | 't5' | 't6';
LABEL : LETTER (LETTER | DIGIT | '_')* ;
IMM : ('+'|'-')?[0-9]+ ;

fragment LETTER : LOWER_LETTER | UPPER_LETTER;
fragment LOWER_LETTER : 'a'..'z';
fragment UPPER_LETTER : 'A'..'Z';
fragment DIGIT : '0'..'9' ;

/* Comments and Useless Characters */
LINE_COMMENT : '//' .*? '\r'? '\n' -> skip; // Match "//" stuff '\n'
COMMENT : '/*' .*? '*/' -> skip; // Match "/*" stuff "*/"
// other comment /home/apr/tools/mappo/dataset/input/dataset_riscv_litmus_repair/HAND/LR-SC-diff-loc1_no_ppo_0.litmus
OTHER_COMMENT : '(*' .*? '*)' -> skip; // Match "/*" stuff "*/"
WS : [ \t\r\n]+ -> skip; // skip spaces, tabs, newlines