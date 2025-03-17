/****************************************************************************/
/*                           the diy toolsuite                              */
/*                                                                          */
/* Jade Alglave, University College London, UK.                             */
/* Luc Maranget, INRIA Paris-Rocquencourt, France.                          */
/*                                                                          */
/* This C source is a product of litmus7 and includes source that is        */
/* governed by the CeCILL-B license.                                        */
/****************************************************************************/

/* Parameters */
#define SIZE_OF_TEST 100
#define NUMBER_OF_RUN 10
#define AVAIL 2
#define STRIDE 1
#define MAX_LOOP 0
#define N 2
#define AFF_INCR (-1)
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

/* params */
typedef struct {
  int verbose;
  int size_of_test,max_run;
  int stride;
  int do_change;
} param_t;


/* Full memory barrier */

inline static void mbar(void) {
  asm __volatile__ ("fence rw,rw" ::: "memory");
}
/**********************/
/* Context definition */
/**********************/


typedef struct {
/* Shared variables */
  int *y;
  int *x;
/* Final content of observed  registers */
  int *out_0_x7;
  int *out_1_x9;
/* Check data */
  pb_t *fst_barrier;
/* Barrier for litmus loop */
/* Instance seed */
  st_t seed;
/* Parameters */
  param_t *_p;
} ctx_t;

inline static int final_cond(int _out_0_x7,int _out_1_x9) {
  switch (_out_0_x7) {
  case 0:
    switch (_out_1_x9) {
    case 2:
      return 1;
    default:
      return 0;
    }
  default:
    return 0;
  }
}

inline static int final_ok(int cond) {
  return cond;
}

/**********************/
/* Outcome collection */
/**********************/
#define NOUTS 2
typedef intmax_t outcome_t[NOUTS];

static const int out_0_x7_f = 0 ;
static const int out_1_x9_f = 1 ;


typedef struct hist_t {
  outs_t *outcomes ;
  count_t n_pos,n_neg ;
} hist_t ;

static hist_t *alloc_hist(void) {
  hist_t *p = malloc_check(sizeof(*p)) ;
  p->outcomes = NULL ;
  p->n_pos = p->n_neg = 0 ;
  return p ;
}

static void free_hist(hist_t *h) {
  free_outs(h->outcomes) ;
  free(h) ;
}

static void add_outcome(hist_t *h, count_t v, outcome_t o, int show) {
  h->outcomes = add_outcome_outs(h->outcomes,o,NOUTS,v,show) ;
}

static void merge_hists(hist_t *h0, hist_t *h1) {
  h0->n_pos += h1->n_pos ;
  h0->n_neg += h1->n_neg ;
  h0->outcomes = merge_outs(h0->outcomes,h1->outcomes,NOUTS) ;
}


static void do_dump_outcome(FILE *fhist, intmax_t *o, count_t c, int show) {
  fprintf(fhist,"%-6"PCTR"%c>0:x7=%i; 1:x9=%i;\n",c,show ? '*' : ':',(int)o[out_0_x7_f],(int)o[out_1_x9_f]);
}

static void just_dump_outcomes(FILE *fhist, hist_t *h) {
  outcome_t buff ;
  dump_outs(fhist,do_dump_outcome,h->outcomes,buff,NOUTS) ;
}

/**************************************/
/* Prefetch (and check) global values */
/**************************************/

static void check_globals(ctx_t *_a) {
  int *y = _a->y;
  int *x = _a->x;
  for (int _i = _a->_p->size_of_test-1 ; _i >= 0 ; _i--) {
    if (rand_bit(&(_a->seed)) && y[_i] != 0) fatal("A, check_globals failed");
    if (rand_bit(&(_a->seed)) && x[_i] != 0) fatal("A, check_globals failed");
  }
  pb_wait(_a->fst_barrier);
}

/***************/
/* Litmus code */
/***************/

typedef struct {
  int th_id; /* I am running on this thread */
  ctx_t *_a;   /* In this context */
} parg_t;

static void *P0(void *_vb) {
  mbar();
  parg_t *_b = (parg_t *)_vb;
  ctx_t *_a = _b->_a;
  check_globals(_a);
  int _size_of_test = _a->_p->size_of_test;
  int _stride = _a->_p->stride;
  int *out_0_x7 = _a->out_0_x7;
  for (int _j = _stride ; _j > 0 ; _j--) {
    int trashed_x5 = 2;

    for (int _i = _size_of_test-_j ; _i >= 0 ; _i -= _stride) {asm __volatile__ (
    "#START _litmus_P0\n"
    "#_litmus_P0_0\n\t"
    "#_litmus_P0_1\n\t"
    "sw %[x5],0(%[x6])\n"
    "#_litmus_P0_2\n\t"
    "lw %[x7],0(%[x8])\n"
    "#END _litmus_P0\n\t"
    :[x7] "=&r" (out_0_x7[_i])
    :[x6] "r" (&_a->x[_i]),[x8] "r" (&_a->y[_i]), [x5] "r" (trashed_x5)
    :"cc","memory"
    );trashed_x5 += 2;

    mbar();
  }
}
mbar();
return NULL;
}

static void *P1(void *_vb) {
  mbar();
  parg_t *_b = (parg_t *)_vb;
  ctx_t *_a = _b->_a;
  check_globals(_a);
  int _size_of_test = _a->_p->size_of_test;
  int _stride = _a->_p->stride;
  int *out_1_x9 = _a->out_1_x9;
  for (int _j = _stride ; _j > 0 ; _j--) {
    int trashed_x5 = 1;

    int trashed_x7 = 1;

    for (int _i = _size_of_test-_j ; _i >= 0 ; _i -= _stride) {asm __volatile__ (
    "#START _litmus_P1\n"
    "#_litmus_P1_0\n\t"
    "#_litmus_P1_1\n\t"
    "sw %[x5],0(%[x6])\n"
    "#_litmus_P1_2\n\t"
    "#_litmus_P1_3\n\t"
    "sw %[x7],0(%[x8])\n"
    "#_litmus_P1_4\n\t"
    "fence rw,rw\n"
    "#_litmus_P1_5\n\t"
    "lw %[x9],0(%[x8])\n"
    "#END _litmus_P1\n\t"
    :[x9] "=&r" (out_1_x9[_i])
    :[x6] "r" (&_a->y[_i]),[x8] "r" (&_a->x[_i]), [x7] "r" (trashed_x7), [x5] "r" (trashed_x5)
    :"cc","memory"
    );trashed_x5 += 2;

    trashed_x7 += 2;

    mbar();
  }
}
mbar();
return NULL;
}

/*******************************************************/
/* Context allocation, freeing and reinitialization    */
/*******************************************************/

static void init(ctx_t *_a) {
  int size_of_test = _a->_p->size_of_test;

  _a->seed = rand();
  _a->out_0_x7 = malloc_check(size_of_test*sizeof(*(_a->out_0_x7)));
  _a->out_1_x9 = malloc_check(size_of_test*sizeof(*(_a->out_1_x9)));
  _a->y = malloc_check(size_of_test*sizeof(*(_a->y)));
  _a->x = malloc_check(size_of_test*sizeof(*(_a->x)));
  _a->fst_barrier = pb_create(N);
}

static void finalize(ctx_t *_a) {
  free((void *)_a->y);
  free((void *)_a->x);
  free((void *)_a->out_0_x7);
  free((void *)_a->out_1_x9);
  pb_free(_a->fst_barrier);
}

static void reinit(ctx_t *_a) {
  for (int _i = _a->_p->size_of_test-1 ; _i >= 0 ; _i--) {
    _a->y[_i] = 0;
    _a->x[_i] = 0;
    _a->out_0_x7[_i] = -239487;
    _a->out_1_x9[_i] = -239487;
  }
}

typedef struct {
  pm_t *p_mutex;
  pb_t *p_barrier;
  param_t *_p;
} zyva_t;

#define NT N

static void *zyva(void *_va) {
  zyva_t *_a = (zyva_t *) _va;
  param_t *_b = _a->_p;
  pb_wait(_a->p_barrier);
  pthread_t thread[NT];
  parg_t parg[N];
  f_t *fun[] = {&P0,&P1};
  hist_t *hist = alloc_hist();
  ctx_t ctx;
  ctx._p = _b;

  init(&ctx);
  for (int _p = N-1 ; _p >= 0 ; _p--) {
    parg[_p].th_id = _p; parg[_p]._a = &ctx;
  }

  for (int n_run = 0 ; n_run < _b->max_run ; n_run++) {
    if (_b->verbose>1) fprintf(stderr,"Run %i of %i\r", n_run, _b->max_run);
    reinit(&ctx);
    if (_b->do_change) perm_funs(&ctx.seed,fun,N);
    for (int _p = NT-1 ; _p >= 0 ; _p--) {
      launch(&thread[_p],fun[_p],&parg[_p]);
    }
    if (_b->do_change) perm_threads(&ctx.seed,thread,NT);
    for (int _p = NT-1 ; _p >= 0 ; _p--) {
      join(&thread[_p]);
    }
    /* Log final states */
    for (int _i = _b->size_of_test-1 ; _i >= 0 ; _i--) {
      int _out_0_x7_i = ctx.out_0_x7[_i];
      int _out_1_x9_i = ctx.out_1_x9[_i];
      outcome_t o;
      int cond;

      cond = final_ok(final_cond(_out_0_x7_i,_out_1_x9_i));
      o[out_0_x7_f] = _out_0_x7_i;
      o[out_1_x9_f] = _out_1_x9_i;
      /* add_outcome(hist,1,o,cond); */
      if (cond) { hist->n_pos++; } else { hist->n_neg++; }
    }
  
int counter_exhaust = 0;
for (int perple_a = 0; perple_a <= _b->size_of_test - 1 ; perple_a++){
  for (int perple_b = 0; perple_b <= _b->size_of_test - 1 ; perple_b++){
    if (ctx.out_1_x9[perple_b] % 2 == 0 && ctx.out_1_x9[perple_b]<=2*(perple_a)+2 && ctx.out_0_x7[perple_a] % 2 == 1 && ctx.out_0_x7[perple_a]<=2*(perple_b-1)+1){
      counter_exhaust += 1;
    }
  }
}
printf("exhaust statistic: %d\n", counter_exhaust);;
}

  finalize(&ctx);
  return hist;
}

#ifdef ASS
static void ass(FILE *out) { }
#else
#include "R.h"
#endif

static void prelude(FILE *out) {
  fprintf(out,"%s\n","%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
  fprintf(out,"%s\n","% Results for /home/whq/Desktop/code_list/perple_test/litmus_test/R.litmus %");
  fprintf(out,"%s\n","%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
  fprintf(out,"%s\n","RISCV A");
  fprintf(out,"%s\n","\"PodWR Fre PodWW Coe\"");
  fprintf(out,"%s\n","{0:x6=x; 0:x8=y; 1:x6=y; 1:x8=x;}");
  fprintf(out,"%s\n"," P0          | P1          ;");
  fprintf(out,"%s\n"," ori x5,x0,2 | ori x5,x0,1 ;");
  fprintf(out,"%s\n"," sw x5,0(x6) | sw x5,0(x6) ;");
  fprintf(out,"%s\n"," lw x7,0(x8) | ori x7,x0,1 ;");
  fprintf(out,"%s\n","             | sw x7,0(x8) ;");
  fprintf(out,"%s\n","             | fence rw,rw ;");
  fprintf(out,"%s\n","             | lw x9,0(x8) ;");
  fprintf(out,"%s\n","");
  fprintf(out,"%s\n","exists (0:x7=0 /\\ 1:x9=2)");
  fprintf(out,"Generated assembler\n");
  ass(out);
}

#define ENOUGH 10

static void postlude(FILE *out,cmd_t *cmd,hist_t *hist,count_t p_true,count_t p_false,tsc_t total) {
  fprintf(out,"Test A Allowed\n");
  fprintf(out,"Histogram (%i states)\n",finals_outs(hist->outcomes));
  just_dump_outcomes(out,hist);
  int cond = p_true > 0;
  fprintf(out,"%s\n",cond?"Ok":"No");
  fprintf(out,"\nWitnesses\n");
  fprintf(out,"Positive: %" PCTR ", Negative: %" PCTR "\n",p_true,p_false);
  fprintf(out,"Condition %s is %svalidated\n","exists (0:x7=0 /\\ 1:x9=2)",cond ? "" : "NOT ");
  fprintf(out,"Hash=b5d039d6d2974678fa3d6e0bfef87798\n");
  fprintf(out,"Generator=diyone7 (version 7.56)\n");
  fprintf(out,"Com=Fr Co\n");
  fprintf(out,"Orig=PodWR Fre PodWW Coe\n");
  count_t cond_true = p_true;
  count_t cond_false = p_false;
  fprintf(out,"Observation A %s %" PCTR " %" PCTR "\n",!cond_true ? "Never" : !cond_false ? "Always" : "Sometimes",cond_true,cond_false);
  if (p_true > 0) {
  }
  fprintf(out,"Time A %.2f\n",total / 1000000.0);
  fflush(out);
}

static void run(cmd_t *cmd,cpus_t *def_all_cpus,FILE *out) {
  if (cmd->prelude) prelude(out);
  tsc_t start = timeofday();
  param_t prm ;
/* Set some parameters */
  prm.verbose = cmd->verbose;
  prm.size_of_test = cmd->size_of_test;
  prm.max_run = cmd->max_run;
  prm.stride = cmd->stride > 0 ? cmd->stride : N ;
  prm.do_change = 1;
  if (cmd->fix) prm.do_change = 0;
/* Computes number of test concurrent instances */
  int n_avail = cmd->avail;
  int n_exe;
  if (cmd->n_exe > 0) {
    n_exe = cmd->n_exe;
  } else {
    n_exe = n_avail < N ? 1 : n_avail / N;
  }
/* Show parameters to user */
  if (prm.verbose) {
    log_error( "A: n=%i, r=%i, s=%i",n_exe,prm.max_run,prm.size_of_test);
    log_error(", st=%i",prm.stride);
    log_error("\n");
  }
  hist_t *hist = NULL;
  int n_th = n_exe-1;
  pthread_t th[n_th];
  zyva_t zarg[n_exe];
  pm_t *p_mutex = pm_create();
  pb_t *p_barrier = pb_create(n_exe);
  for (int k=0 ; k < n_exe ; k++) {
    zyva_t *p = &zarg[k];
    p->_p = &prm;
    p->p_mutex = p_mutex; p->p_barrier = p_barrier; 
    if (k < n_th) {
      launch(&th[k],zyva,p);
    } else {
      hist = (hist_t *)zyva(p);
    }
  }

  count_t n_outs = prm.size_of_test; n_outs *= prm.max_run;
  for (int k=0 ; k < n_th ; k++) {
    hist_t *hk = (hist_t *)join(&th[k]);
    if (sum_outs(hk->outcomes) != n_outs || hk->n_pos + hk->n_neg != n_outs) {
/* fatal("A, sum_hist") */
    }
    merge_hists(hist,hk);
    free_hist(hk);
  }
  tsc_t total = timeofday() - start;
  pm_free(p_mutex);
  pb_free(p_barrier);

  n_outs *= n_exe ;
  if (sum_outs(hist->outcomes) != n_outs || hist->n_pos + hist->n_neg != n_outs) {
/* fatal("A, sum_hist") */
  }
  count_t p_true = hist->n_pos, p_false = hist->n_neg;
  postlude(out,cmd,hist,p_true,p_false,total);
  free_hist(hist);
}


int A(int argc, char **argv, FILE *out) {
  cpus_t *def_all_cpus = NULL;
  cmd_t def = { 0, NUMBER_OF_RUN, SIZE_OF_TEST, STRIDE, AVAIL, 0, 0, aff_none, 0, 0, AFF_INCR, def_all_cpus, NULL, -1, MAX_LOOP, NULL, NULL, -1, -1, -1, 0, 1};
  cmd_t cmd = def;
  parse_cmd(argc,argv,&def,&cmd);
  run(&cmd,def_all_cpus,out);
  return EXIT_SUCCESS;
}
