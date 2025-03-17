
/**********************/
/* Context definition */
/**********************/


typedef struct {
/* Shared variables */
@VARS@
/* Final content of observed  registers */
@REGS@
/* Check data */
  pb_t *fst_barrier;
/* Barrier for litmus loop */
  int volatile *barrier;
/* Instance seed */
  st_t seed;
/* Parameters */
  param_t *_p;
} ctx_t;

inline static int final_cond(int a, ...) {
  return 0;
}

inline static int final_ok(int cond) {
  return cond;
}
