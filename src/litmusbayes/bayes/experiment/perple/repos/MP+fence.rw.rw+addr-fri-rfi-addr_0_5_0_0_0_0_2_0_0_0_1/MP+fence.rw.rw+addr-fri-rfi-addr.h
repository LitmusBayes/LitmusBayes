static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a7,0(a5)");
  fprintf(out,"%s\n","	xor s9,a7,a7");
  fprintf(out,"%s\n","	add s8,a4,s9");
  fprintf(out,"%s\n","	lw t1,0(s8)");
  fprintf(out,"%s\n","	sw a2,0(a4)");
  fprintf(out,"%s\n","	lw t3,0(a4)");
  fprintf(out,"%s\n","	xor s7,t3,t3");
  fprintf(out,"%s\n","	add s6,a3,s7");
  fprintf(out,"%s\n","	lw s5,0(s6)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	fence rw,rw");
  fprintf(out,"%s\n","	sw a2,0(a4)");
}
