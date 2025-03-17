static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a7,0(a5)");
  fprintf(out,"%s\n","	xor t4,a7,a7");
  fprintf(out,"%s\n","	ori t4,t4,1");
  fprintf(out,"%s\n","	sw t4,0(a4)");
  fprintf(out,"%s\n","	#_litmus_P1_5");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	lw a6,0(a5)");
  fprintf(out,"%s\n","	fence r,rw");
  fprintf(out,"%s\n","	#_litmus_P0_3");
}
