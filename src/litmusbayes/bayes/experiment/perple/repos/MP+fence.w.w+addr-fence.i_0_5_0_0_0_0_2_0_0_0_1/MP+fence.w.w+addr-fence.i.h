static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw t1,0(a5)");
  fprintf(out,"%s\n","	xor t0,t1,t1");
  fprintf(out,"%s\n","	add t5,a4,t0");
  fprintf(out,"%s\n","	lw t6,0(t5)");
  fprintf(out,"%s\n","	fence.i");
  fprintf(out,"%s\n","	lw t3,0(a3)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	#_litmus_P0_1");
  fprintf(out,"%s\n","	fence w,w");
  fprintf(out,"%s\n","	#_litmus_P0_4");
}
