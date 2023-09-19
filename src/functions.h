#pragma once


#include <stdlib.h>



struct kmeans_args_t {
  int*               input_vals;
  int                n_cluster;
  int                dims;
  int                max_iter;
  double             threshold;
  bool               c_flag;
  int                seed;
  
};

prefix_sum_args_t* alloc_args(int n_threads);

/*int next_power_of_two(int x);

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
               my_barrier *counter_barrier);
               */
