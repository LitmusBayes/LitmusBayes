import os
import re

from src.litmusbayes import config
from src.litmusbayes.perple.perple import perpLE
from src.litmusbayes.utils.file_util import search_file, read_file


temp = '''
#define CACHELINE 64

static volatile int stress_stop = 0;

typedef struct {
  int cpu_id;        // 可选：绑定到某个core
  size_t region_sz; 
  int mode;          // 0=store buffer, 1=coherence-pingpong, 2=mixed
  volatile int *buf; 
  volatile int *buf2; // 用于 ping-pong
} stress_arg_t;

// ==========================================================
// 1. Store Buffer 压力：不断写大量 distinct 地址
// ==========================================================
void *stress_worker_store_buffer(void *arg) {
  volatile uint64_t *buf = aligned_alloc(64, 4 * 1024 * 1024);
  size_t sz = (4 * 1024 * 1024) / sizeof(uint64_t);

  while (!stress_stop) {
      for (size_t i = 0; i < sz; i += 8) {
          buf[i] = i;    // 连续 stores
      }
  }
    return NULL;
}

void* stress_store_buffer(void *arg0){
  stress_arg_t *arg = (stress_arg_t*)arg0;
  volatile int *buf = arg->buf;
  size_t sz = arg->region_sz;

  // 连续 store，制造 SB 饱和
  while (!stress_stop) {
      for (size_t i = 0; i < sz; i+=64/sizeof(int)) {
          buf[i] = i;            // 覆盖大量cache line
      }
  }
}


// ==========================================================
// 2. coherence ping-pong：多线程写同一个 line
// ==========================================================
void *stress_worker_pingpong(void *shared) {
    volatile uint64_t *p = (uint64_t *)shared;
    while (!stress_stop) {
        (*p)++;        // 多线程共享同一 cache line → coherence 震荡
    }
    return NULL;
}
void* stress_pingpong(void *arg0){
  stress_arg_t *arg = (stress_arg_t*)arg0;
  volatile int *x = arg->buf;
  volatile int *y = arg->buf2;

  while (!stress_stop) {
      *x = 1;     // 核A写 → M
      *y = *x;    // 核B读 → S，产生 invalidation
      *x = 2;     // 再写 → 再次 invalidation
      *y = *x;    // load again
  }
}
// ==========================================================
// 3. 统一启动 stress 线程
// ==========================================================
#define NSTRESS 4

pthread_t stress_th[NSTRESS];
stress_arg_t sarg[NSTRESS];
void start_stress_threads() {
  for (int i = 0; i < NSTRESS; i++) {

      sarg[i].region_sz = 4096 * 4;
      sarg[i].buf  = aligned_alloc(64, sarg[i].region_sz);
      sarg[i].buf2 = aligned_alloc(64, sarg[i].region_sz);

      memset((void*)sarg[i].buf, 0, sarg[i].region_sz);
      memset((void*)sarg[i].buf2, 0, sarg[i].region_sz);

      int mode = i % 2; // 一半 store buffer，一半 ping-pong

      if (mode == 0)
          pthread_create(&stress_th[i], NULL, stress_store_buffer, &sarg[i]);
      else
          pthread_create(&stress_th[i], NULL, stress_pingpong, &sarg[i]);
  }
}


pthread_t *stress_start(int n_storebuf, size_t sb_size,
                        int n_pingpong)
{
    int total = n_storebuf + n_pingpong;
    pthread_t *th = malloc(sizeof(pthread_t) * total);

    volatile uint64_t *shared_line =
        aligned_alloc(CACHELINE, CACHELINE);

    int k = 0;

    // 启动 store-buffer 压力线程
    for (int i = 0; i < n_storebuf; i++)
        pthread_create(&th[k++], NULL,
            stress_worker_store_buffer, (void*)sb_size);

    // 启动 ping-pong 线程
    for (int i = 0; i < n_pingpong; i++)
        pthread_create(&th[k++], NULL,
            stress_worker_pingpong,
            (void*)shared_line);

    return th;
}

void stress_stop_all(pthread_t *th, int total) {
    stress_stop = 1;
    for (int i = 0; i < total; i++)
        pthread_join(th[i], NULL);
}
'''

def add_memory_stress(input_file, output_file):
    lines = []
    new_lines = []
    with open(input_file, "r") as f:
        lines = f.readlines()
    include_flag = False
    start_flag = False
    for line in lines:
        if "#include" in line and not include_flag:
            new_lines.append("#include <string.h>\n")
            include_flag = True
        if "static void run" in line:
            start_flag = True
            new_lines.append(temp)
        new_lines.append(line)
        if start_flag:
            start_flag = False
            new_lines.append("  if (!stress_started) {\n")
            new_lines.append("start_stress_threads();\n")
            new_lines.append("stress_started = 1;\n")
            new_lines.append("}\n")
    with open(output_file, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":

    litmus_path_dir = [
        # ("PPOCA.c", "PPOCA_change.c", f'{config.TEST_DIR}/experiment/exp1_litmus/PPOCA.litmus'),
        # ("R.c", "R_change.c","R_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/R.litmus'),
        # ("SB.c", "SB_change.c", "SB_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/SB.litmus'),
        # ("LB.c", "LB_change.c", "LB_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/LB.litmus'),
        # ("MP.c", "MP_change.c", "MP_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/MP.litmus'),
        # ("S.c", "S_change.c", "S_change_h.c", f'/home/whq/Desktop/code_list/perple_test/litmus_test/S.litmus'),
        ("/home/whq/Desktop/code_list/perple_test/stress/SB/SB.c","/home/whq/Desktop/code_list/perple_test/stress/SB/SB.c"),

    ]

    # for input_path, output_path in litmus_path_dir:
    #     add_memory_stress(input_path, output_path)
    test_dir = '/home/whq/Desktop/code_list/perple_test'
    litmus_dir = os.path.join(test_dir, 'litmus_test')
    files = [f for f in os.listdir(litmus_dir) if os.path.isfile(os.path.join(litmus_dir, f))]
    arg_dir = os.path.join(test_dir, 'stress/arg_perple')

    for root, dirs, files in os.walk(arg_dir):
        for d in dirs:
            if d == "log":
                continue
            litmus_path = os.path.join(root, d)
            litmus_name = d.split("_")[0]
            mode = "_".join(d.split("_")[1:])
            print(litmus_name, mode, iter)
            # if os.path.exists(os.path.join(chip_dir, f"{d}/{d}-100000-{0}")):
            #     continue
            add_memory_stress(f"{litmus_path}/{litmus_name}.c", f"{litmus_path}/{litmus_name}.c")

