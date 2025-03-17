static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a6,0(a4)");
  fprintf(out,"%s\n","	sw a3,0(a4)");
  fprintf(out,"%s\n","	lw a7,0(a4)");
  fprintf(out,"%s\n","	xor t2,a7,a7");
  fprintf(out,"%s\n","	add a5,a5,t2");
  fprintf(out,"%s\n","	lw t1,0(a5)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a4)");
  fprintf(out,"%s\n","	fence w,w");
  fprintf(out,"%s\n","	sw a2,0(a5)");
}
