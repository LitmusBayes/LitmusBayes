static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	sw a3,0(a5)");
  fprintf(out,"%s\n","	lw t5,0(a4)");
  fprintf(out,"%s\n","	xor t4,t5,t5");
  fprintf(out,"%s\n","	add t3,a4,t4");
  fprintf(out,"%s\n","	lw a6,0(t3)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	lw a6,0(a4)");
}
