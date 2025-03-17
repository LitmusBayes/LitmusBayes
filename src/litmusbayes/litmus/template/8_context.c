/*******************************************************/
/* Context allocation, freeing and reinitialization    */
/*******************************************************/

static void init(ctx_t *_a) {
  int size_of_test = _a->_p->size_of_test;

  _a->seed = rand();
@INIT_REG@
@INIT_VAR@
  _a->fst_barrier = pb_create(N);
  _a->s_or = po_create(N);
  for (int _p = N-1 ; _p >= 0 ; _p--) {
@INIT_COPY_VAR@
  }
  _a->barrier = malloc_check(size_of_test*sizeof(*(_a->barrier)));
}

static void finalize(ctx_t *_a) {
@FREE_VAR@
@FREE_REG@
  pb_free(_a->fst_barrier);
  po_free(_a->s_or);
  for (int _p = N-1 ; _p >= 0 ; _p--) {
@FREE_COPY_VAR@
  }
  free((void *)_a->barrier);
}

static void reinit(ctx_t *_a) {
  for (int _i = _a->_p->size_of_test-1 ; _i >= 0 ; _i--) {
@REINIT_VAR@
@REINIT_REG@
    _a->barrier[_i] = 0;
  }
}
