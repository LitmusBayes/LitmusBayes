/*
 Topology: {{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}}}
*/

static int cpu_scan[] = {
// [[0],[1]]
12, 0, 6, 2, 10, 4, 3, 13, 14, 5, 1, 8, 11, 7, 9, 15,
// [[0,1]]
2, 3, 15, 14, 1, 0, 13, 12, 10, 11, 7, 6, 9, 8, 5, 4,
};

static char *group[] = {
"[[0],[1]]",
"[[0,1]]",
};

#define SCANSZ 2
#define SCANLINE 16

static count_t ngroups[SCANSZ];