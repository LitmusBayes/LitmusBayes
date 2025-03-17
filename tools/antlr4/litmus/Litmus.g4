grammar Litmus;
options { language=Python3; }

entry: init location? filter? final EOF ;
init: '{' (var_decl';' | reg_decl';' | reg_cond';' | var_cond';' | addr_cond';') * '}' ;
var_decl: VAR_TYPE (VAR | POINTER) ('=' ADDR_OF?VAR)?;
reg_decl: VAR_TYPE PID REG;
final: NOT?QUANTIFIER cond_expr;
location: 'locations' '[' observed_var* ']';
filter: 'filter' cond_expr;
observed_var: (PID REG ';') | (VAR ';');
cond_expr: cond_term (OP cond_term)* ;
cond_term : cond | NOT?'(' cond_expr ')' ;
// include addr_cond into cond to reach the case: ISA-S-DEP-ADDR-SUCCESS
cond:  addr_cond | var_cond | reg_cond | 'true' | 'false';
reg_cond: PID REG '=' IMM;
var_cond: VAR '=' IMM;
addr_cond: PID REG '=' VAR;

ADDR_OF: '&';
NOT: '~' | 'not';
PID: DIGIT':';
REG : 'x0' | 'x1' | 'x2' | 'x3' | 'x4' | 'x5' | 'x6' | 'x7' | 'x8' | 'x9' | 'x10' | 'x11' | 'x12' | 'x13' | 'x14' | 'x15' | 'x16' | 'x17' | 'x18' | 'x19' | 'x20' | 'x21' | 'x22' | 'x23' | 'x24' | 'x25' | 'x26' | 'x27' | 'x28' | 'x29' | 'x30' | 'x31' | 'zero' | 'ra' | 'sp' | 'gp' | 'tp' | 't0' | 't1' | 't2' | 's0' | 's1' | 'a0' | 'a1' | 'a2' | 'a3' | 'a4' | 'a5' | 'a6' | 'a7' | 's2' | 's3' | 's4' | 's5' | 's6' | 's7' | 's8' | 's9' | 's10' | 's11' | 't3' | 't4' | 't5' | 't6';
IMM : ('+'|'-')?[0-9]+ ;
OP: '/\\' | '\\/';
QUANTIFIER: 'exists' | 'forall' ;
VAR_TYPE: 'int64_t' | 'uint64_t' ;
VAR: LETTER(LETTER | DIGIT | '_')* ;
POINTER: '*'VAR;



fragment LETTER : LOWER_LETTER | UPPER_LETTER;
fragment LOWER_LETTER : 'a'..'z';
fragment UPPER_LETTER : 'A'..'Z';
fragment DIGIT : '0'..'9' ;

/* Comments and Useless Characters */
LINE_COMMENT : '//' .*? '\r'? '\n' -> skip; // Match "//" stuff '\n'
COMMENT : '/*' .*? '*/' -> skip; // Match "/*" stuff "*/"
WS : [ \t\r\n]+ -> skip; // skip spaces, tabs, newlines
