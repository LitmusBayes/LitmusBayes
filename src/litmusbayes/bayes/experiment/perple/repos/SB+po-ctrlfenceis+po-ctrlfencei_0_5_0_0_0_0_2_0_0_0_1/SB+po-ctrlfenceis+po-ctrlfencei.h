static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	lw t4,0(a4)");
  fprintf(out,"%s\n","	bne t4,x0,0f");
  fprintf(out,"%s\n","	0:");
  fprintf(out,"%s\n","	fence.i");
  fprintf(out,"%s\n","	lw a7,0(a3)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a3,0(a5)");
  fprintf(out,"%s\n","	lw t3,0(a4)");
  fprintf(out,"%s\n","	bne t3,x0,0f");
  fprintf(out,"%s\n","	0:");
  fprintf(out,"%s\n","	fence.i");
  fprintf(out,"%s\n","	lw a6,0(a4)");
}
