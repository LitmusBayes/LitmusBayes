/**************************************/
/* Prefetch (and check) global values */
/**************************************/

static void check_globals(ctx_t *_a) {
@INIT_GLOBALS@
  for (int _i = _a->_p->size_of_test-1 ; _i >= 0 ; _i--) {
@CHECK_GLOBALS@
  }
  pb_wait(_a->fst_barrier);
}


/* We omit stabilize_globals here */