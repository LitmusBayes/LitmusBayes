#include <stdio.h>
#include <stdlib.h>

/* Declarations of tests entry points */
extern int A(int argc,char **argv,FILE *out);

/* Date function */
#include <time.h>
static void my_date(FILE *out) {
  time_t t = time(NULL);
  fprintf(out,"%s",ctime(&t));
}

/* Postlude */
static void end_report(int argc,char **argv,FILE *out) {
  fprintf(out,"%s\n","Revision exported, version 7.56");
  fprintf(out,"%s\n","Command line: litmus7 -carch RISCV -limit true -affinity incr1 -force_affinity true -mem direct -barrier userfence -detached false -thread std -launch changing -stride 0 -size_of_test 100 -number_of_run 10 -driver C -gcc riscv64-unknown-linux-gnu-gcc -ccopts -O2 -linkopt -static -avail 4 -alloc dynamic -contiguous false -noalign none ./make_new_litmus/litmus_output/Rfe_DpAddrsR_Fre_PodWR_Fre_PodWW.litmus -o ./make_new_litmus/litmus_output_bayes/Rfe_DpAddrsR_Fre_PodWR_Fre_PodWW_0_1_0_0_0_0_2_0_0_0_0");
  fprintf(out,"%s\n","Parameters");
  fprintf(out,"%s\n","#define SIZE_OF_TEST 100");
  fprintf(out,"%s\n","#define NUMBER_OF_RUN 10");
  fprintf(out,"%s\n","#define AVAIL 4");
  fprintf(out,"%s\n","#define STRIDE (-1)");
  fprintf(out,"%s\n","#define MAX_LOOP 0");
  fprintf(out,"%s\n","/* gcc options: -D_GNU_SOURCE -DFORCE_AFFINITY -Wall -std=gnu99 -O2 -pthread */");
  fprintf(out,"%s\n","/* gcc link options: -static */");
  fprintf(out,"%s\n","/* barrier: userfence */");
  fprintf(out,"%s\n","/* launch: changing */");
  fprintf(out,"%s\n","/* affinity: incr1 */");
  fprintf(out,"%s\n","/* alloc: dynamic */");
  fprintf(out,"%s\n","/* memory: direct */");
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
  A(argc,argv,out);
  end_report(argc,argv,out);
  my_date(out);
}

int main(int argc,char **argv) {
  run(argc,argv,stdout);
  return 0;
}
