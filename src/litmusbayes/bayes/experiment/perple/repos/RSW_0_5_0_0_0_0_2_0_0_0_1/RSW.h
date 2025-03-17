static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw t1,0(a5)");
  fprintf(out,"%s\n","	xor s2,t1,t1");
  fprintf(out,"%s\n","	add t2,a4,s2");
  fprintf(out,"%s\n","	lw s1,0(t2)");
  fprintf(out,"%s\n","	lw t0,0(a4)");
  fprintf(out,"%s\n","	xor t6,t0,t0");
  fprintf(out,"%s\n","	add t5,a3,t6");
  fprintf(out,"%s\n","	lw t3,0(t5)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	#_litmus_P0_1");
  fprintf(out,"%s\n","	fence rw,rw");
  fprintf(out,"%s\n","	#_litmus_P0_4");
}
