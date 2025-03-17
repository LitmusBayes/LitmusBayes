static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P2");
  fprintf(out,"%s\n","	lw a7,0(a5)");
  fprintf(out,"%s\n","	bne a7,x0,0f");
  fprintf(out,"%s\n","	0:");
  fprintf(out,"%s\n","	fence.i");
  fprintf(out,"%s\n","	lw t1,0(a4)");
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a6,0(a5)");
  fprintf(out,"%s\n","	xor t4,a6,a6");
  fprintf(out,"%s\n","	add t3,a4,t4");
  fprintf(out,"%s\n","	sw a3,0(t3)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a2,0(a5)");
  fprintf(out,"%s\n","	fence rw,rw");
  fprintf(out,"%s\n","	sw a2,0(a4)");
}
