static void ass(FILE *out) {
  fprintf(out,"%s\n","#START _litmus_P0");
  fprintf(out,"%s\n","	lw a4,0(a5)");
  fprintf(out,"%s\n","	xor t0,a4,a4");
  fprintf(out,"%s\n","	add t6,a5,t0");
  fprintf(out,"%s\n","	lw a1,0(t6)");
  fprintf(out,"%s\n","#START _litmus_P2");
  fprintf(out,"%s\n","	ori t3,x0,1");
  fprintf(out,"%s\n","	sw t3,0(a5)");
  fprintf(out,"%s\n","	ori a2,x0,1");
  fprintf(out,"%s\n","	sw a2,0(a4)");
  fprintf(out,"%s\n","#START _litmus_P1");
  fprintf(out,"%s\n","	ori t4,x0,2");
  fprintf(out,"%s\n","	sw t4,0(a5)");
  fprintf(out,"%s\n","	lw a1,0(a4)");
}
