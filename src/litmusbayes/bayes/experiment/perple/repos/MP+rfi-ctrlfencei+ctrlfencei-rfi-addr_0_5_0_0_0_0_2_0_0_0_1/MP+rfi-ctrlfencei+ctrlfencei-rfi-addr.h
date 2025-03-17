static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a7,0(a5)");
  fprintf(out,"%s\n","	bne a7,x0,0f");
  fprintf(out,"%s\n","	0:");
  fprintf(out,"%s\n","	fence.i");
  fprintf(out,"%s\n","	sw a2,0(a4)");
  fprintf(out,"%s\n","	lw t1,0(a4)");
  fprintf(out,"%s\n","	xor s2,t1,t1");
  fprintf(out,"%s\n","	add s1,a3,s2");
  fprintf(out,"%s\n","	lw t3,0(s1)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	lw a6,0(a5)");
  fprintf(out,"%s\n","	bne a6,x0,0f");
  fprintf(out,"%s\n","	0:");
  fprintf(out,"%s\n","	fence.i");
  fprintf(out,"%s\n","	sw a2,0(a4)");
}
