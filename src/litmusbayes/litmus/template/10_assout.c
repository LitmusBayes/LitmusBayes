#ifdef ASS
static void ass(FILE *out) { }
#else
#include "@NAME@.h"
#endif

static void prelude(FILE *out) {
  fprintf(out,"%s\n","%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
  fprintf(out,"%s\n","% Results for @NAME@.litmus %");
  fprintf(out,"%s\n","%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
@LITMUS@
  fprintf(out,"Generated assembler\n");
  ass(out);
}
