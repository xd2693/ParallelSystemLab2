#include "helpers.h"

prefix_sum_args_t* alloc_args(int n_threads) {
  return (prefix_sum_args_t*) malloc(n_threads * sizeof(prefix_sum_args_t));
}

int next_power_of_two(int x) {
    int pow = 1;
    while (pow < x) {
        pow *= 2;
    }
    return pow;
}

void fill_args(prefix_sum_args_t *args,
               int n_threads,
               int n_vals,
               int *inputs,
               int *outputs,
               int *sum_offsets,
               bool spin,
               int (*op)(int, int, int),
               int n_loops,
               pthread_barrier_t *barrier,
               my_barrier *counter_barrier) {
    for (int i = 0; i < n_threads; ++i) {
        args[i] = {inputs, outputs, sum_offsets, spin, n_vals,
                   n_threads, i, op, n_loops, barrier, counter_barrier};
    }
}