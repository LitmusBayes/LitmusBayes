static void ass(FILE *out) {
  fprintf(out,"%s\n","	#START _litmus_P1");
  fprintf(out,"%s\n","	sd a3,0(a5)");
  fprintf(out,"%s\n","	ld t1,0(a5)");
  fprintf(out,"%s\n","	ld t3,0(a4)");
  fprintf(out,"%s\n","	#START _litmus_P0");
  fprintf(out,"%s\n","	sd a3,0(a5)");
  fprintf(out,"%s\n","	ld t1,0(a5)");
  fprintf(out,"%s\n","	ld t3,0(a4)");
}
