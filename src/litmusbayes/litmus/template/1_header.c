/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include "utils.h"
#include "outs.h"
#include "affinity.h"

/* params */
typedef struct {
  int verbose;
  int size_of_test,max_run;
  int stride;
  aff_mode_t aff_mode;
  int ncpus, ncpus_used;
  int do_change;
  cpus_t *cm;
} param_t;


/* Full memory barrier */

inline static void mbar(void) {
  asm __volatile__ ("fence rw,rw" ::: "memory");
}

/* Barriers macros */
inline static void barrier_wait(unsigned int id, unsigned int k, int volatile *b) {
  if ((k % N) == id) {
    *b = 1 ;
    mbar();
  } else {
    while (*b == 0) ;
  }
}
