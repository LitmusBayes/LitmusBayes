#include <stdio.h>
#include <stdlib.h>

/* Declarations of tests entry points */
extern int SB_2B_fence_2E_rw_2E_rw_2B_po_2D_addr(int argc,char **argv,FILE *out);

/* Date function */
#include <time.h>
static void my_date(FILE *out) {
  time_t t = time(NULL);
  fprintf(out,"%s",ctime(&t));
}

/* Postlude */
static void end_report(int argc,char **argv,FILE *out) {
  fprintf(out,"%s\n","Revision exported, version 7.56");
  fprintf(out,"%s\n","Command line: litmus7 -carch RISCV -limit true -affinity incr1 -force_affinity true -mem direct -barrier none -detached false -thread std -launch changing -stride 1 -size_of_test 100 -number_of_run 10 -driver C -gcc riscv64-unknown-linux-gnu-gcc -ccopts -O2 -linkopt -static -smtmode seq -smt 2 -avail 4 -alloc dynamic -contiguous false -noalign none /home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive_perple/SB+fence.rw.rw+po-addr.litmus -o /home/whq/Desktop/code_list/perple_test/bayes_banana_perple/SB+fence.rw.rw+po-addr_0_5_0_0_0_0_2_0_0_0_1");
  fprintf(out,"%s\n","Parameters");
  fprintf(out,"%s\n","#define SIZE_OF_TEST 100");
  fprintf(out,"%s\n","#define NUMBER_OF_RUN 10");
  fprintf(out,"%s\n","#define AVAIL 4");
  fprintf(out,"%s\n","#define STRIDE 1");
  fprintf(out,"%s\n","#define MAX_LOOP 0");
  fprintf(out,"%s\n","/* gcc options: -D_GNU_SOURCE -DFORCE_AFFINITY -Wall -std=gnu99 -O2 -pthread */");
  fprintf(out,"%s\n","/* gcc link options: -static */");
  fprintf(out,"%s\n","/* barrier: none */");
  fprintf(out,"%s\n","/* launch: changing */");
  fprintf(out,"%s\n","/* affinity: incr1 */");
  fprintf(out,"%s\n","/* alloc: dynamic */");
  fprintf(out,"%s\n","/* memory: direct */");
  fprintf(out,"%s\n","/* stride: 1 */");
  fprintf(out,"%s\n","/* safer: write */");
  fprintf(out,"%s\n","/* preload: random */");
  fprintf(out,"%s\n","/* speedcheck: no */");
  fprintf(out,"%s\n","/* proc used: 4 */");
/* Command line options */
  fprintf(out,"Command:");
  for ( ; *argv ; argv++) {
    fprintf(out," %s",*argv);
  }
  putc('\n',out);
}

/* Run all tests */
static void run(int argc,char **argv,FILE *out) {
  my_date(out);
  SB_2B_fence_2E_rw_2E_rw_2B_po_2D_addr(argc,argv,out);
  end_report(argc,argv,out);
  my_date(out);
}

int main(int argc,char **argv) {
  run(argc,argv,stdout);
  return 0;
}
