static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	lw t0,0(a4)");
  fprintf(out,"%s\n","	xor t6,t0,t0");
  fprintf(out,"%s\n","	add t5,a3,t6");
  fprintf(out,"%s\n","	lw a7,0(t5)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	lw t6,0(a4)");
  fprintf(out,"%s\n","	xor t5,t6,t6");
  fprintf(out,"%s\n","	add t4,a3,t5");
  fprintf(out,"%s\n","	lw a7,0(t4)");
}
