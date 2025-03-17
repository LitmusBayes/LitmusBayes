static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	lw a6,0(a5)");
  fprintf(out,"%s\n","	xor s2,a6,a6");
  fprintf(out,"%s\n","	ori s2,s2,1");
  fprintf(out,"%s\n","	sw s2,0(a4)");
  fprintf(out,"%s\n","	lw a7,0(a4)");
  fprintf(out,"%s\n","	xor s1,a7,a7");
  fprintf(out,"%s\n","	add t2,a3,s1");
  fprintf(out,"%s\n","	lw t1,0(t2)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sw a3,0(a5)");
  fprintf(out,"%s\n","	lw a6,0(a5)");
  fprintf(out,"%s\n","	xor t4,a6,a6");
  fprintf(out,"%s\n","	add t3,a4,t4");
  fprintf(out,"%s\n","	sw a3,0(t3)");
}
