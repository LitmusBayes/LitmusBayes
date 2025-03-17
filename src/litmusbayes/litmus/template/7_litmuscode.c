static void *P@PID@(void *_vb) {
  mbar();
  parg_t *_b = (parg_t *)_vb;
  ctx_t *_a = _b->_a;
  int _ecpu = _b->cpu[_b->th_id];
  force_one_affinity(_ecpu,AVAIL,_a->_p->verbose,"@NAME@");
  check_globals(_a);
  int _th_id = _b->th_id;
  int volatile *barrier = _a->barrier;
  int _size_of_test = _a->_p->size_of_test;
  int _stride = _a->_p->stride;
@DECL_OUT_REG@
  for (int _j = _stride ; _j > 0 ; _j--) {
    for (int _i = _size_of_test-_j ; _i >= 0 ; _i -= _stride) {
      barrier_wait(_th_id,_i,&barrier[_i]);
@DECL_TRASH_REG@
asm __volatile__ (
"\n"
"#START _litmus_P@PID@\n"
@ASM@
"#END _litmus_P@PID@\n\t"
@OUT_REGS@
@IN_REGS@
:"cc","memory"
);
    }
  }
//  stabilize_globals(0,_a);
  mbar();
  return NULL;
}