#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

_Atomic int y = 0;
_Atomic int x = 0;

void* thread_P0(void* _){
  int r0 = atomic_load_explicit(y, memory_order_relaxed);
  atomic_store_explicit(x, 1, memory_order_release);
  return NULL;
}

void* thread_P1(void* _){
  int r1 = atomic_load_explicit(x, memory_order_acquire);
  if (r1) {
  y = 1;
  }
  return NULL;
}

static inline void reset_globals(){
  atomic_store_explicit(&y, 0, memory_order_relaxed);
  atomic_store_explicit(&x, 0, memory_order_relaxed);
}

static inline int check_exists(){
  int y_snap = atomic_load_explicit(&y, memory_order_relaxed);
  int x_snap = atomic_load_explicit(&x, memory_order_relaxed);
  return (x_snap==1 && y_snap==1) ? 1 : 0;
}

int main(int argc, char** argv){
  long iters = 100000;
  long count = 0;
  pthread_t tids[2];
  for (long it=0; it<iters; ++it){
    reset_globals();
    pthread_create(&tids[0], NULL, thread_P0, NULL);
    pthread_create(&tids[1], NULL, thread_P1, NULL);
    pthread_join(tids[0], NULL);
    pthread_join(tids[1], NULL);
    count += check_exists();
  }
  printf("exists observed: %ld / %ld\n", count, iters);
  return 0;
}
