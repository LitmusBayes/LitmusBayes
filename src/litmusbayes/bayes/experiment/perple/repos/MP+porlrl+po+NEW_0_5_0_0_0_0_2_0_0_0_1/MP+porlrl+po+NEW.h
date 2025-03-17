static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a7,0(a5)");
  fprintf(out,"%s\n","	lw t1,0(a4)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	amoswap.w.rl x0,a2,(a5)");
  fprintf(out,"%s\n","	amoswap.w.rl x0,a2,(a4)");
}
