#define ENOUGH 10

static void postlude(FILE *out,cmd_t *cmd,hist_t *hist,count_t p_true,count_t p_false,tsc_t total) {
  fprintf(out,"Test @NAME@ Allowed\n");
  fprintf(out,"Histogram (%i states)\n",finals_outs(hist->outcomes));
  just_dump_outcomes(out,hist);
  int cond = p_true > 0;
  fprintf(out,"%s\n",cond?"Ok":"No");
  fprintf(out,"\nWitnesses\n");
  fprintf(out,"Positive: %" PCTR ", Negative: %" PCTR "\n",p_true,p_false);
  fprintf(out,"Condition %s is %svalidated\n","@COND@",cond ? "" : "NOT ");
  fprintf(out,"Hash=@HASH@\n");
  count_t cond_true = p_true;
  count_t cond_false = p_false;
  fprintf(out,"Observation @NAME@ %s %" PCTR " %" PCTR "\n",!cond_true ? "Never" : !cond_false ? "Always" : "Sometimes",cond_true,cond_false);
  if (p_true > 0) {
    if (cmd->aff_mode == aff_scan) {
      for (int k = 0 ; k < SCANSZ ; k++) {
        count_t c = ngroups[k];
        if ((c*100)/p_true > ENOUGH) { printf("Topology %-6" PCTR":> %s\n",c,group[k]); }
      }
    } else if (cmd->aff_mode == aff_topo) {
      printf("Topology %-6" PCTR ":> %s\n",ngroups[0],cmd->aff_topo);
    }
  }
  fprintf(out,"Time @NAME@ %.2f\n",total / 1000000.0);
  fflush(out);
}