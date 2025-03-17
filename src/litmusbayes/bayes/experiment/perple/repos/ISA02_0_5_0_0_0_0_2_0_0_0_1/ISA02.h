static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	#_litmus_P1_1");
  fprintf(out,"%s\n","	lw t1,0(a4)");
  fprintf(out,"%s\n","	fence r,r");
  fprintf(out,"%s\n","	lw t3,0(a5)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	#_litmus_P0_1");
  fprintf(out,"%s\n","	lw t1,0(a5)");
  fprintf(out,"%s\n","	fence r,r");
  fprintf(out,"%s\n","	lw t3,0(a4)");
}
