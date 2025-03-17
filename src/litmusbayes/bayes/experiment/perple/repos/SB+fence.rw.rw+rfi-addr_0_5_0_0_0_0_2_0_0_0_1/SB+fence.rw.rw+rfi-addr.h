static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	sw a3,0(a5)");
  fprintf(out,"%s\n","	lw t1,0(a5)");
  fprintf(out,"%s\n","	xor t6,t1,t1");
  fprintf(out,"%s\n","	add t5,a4,t6");
  fprintf(out,"%s\n","	lw t3,0(t5)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	fence rw,rw");
  fprintf(out,"%s\n","	lw a6,0(a4)");
}
