#pragma once
#include <cmath>
#include <cfloat>
#include "argparse.h"

const int SHARED_MEMORY_BYTES = 48 * (1 << 10); //48KB shared mem for target T4 

void wrapper_new_centers(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int blocks,
                         int threads);

__global__ void new_centers(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c);



void wrapper_new_centers_shmem(double *input_vals_c, 
                         double *centers_c,
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int threads);
                         
__global__ void new_centers_shmem(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c);

void wrapper_new_centers_shful(double *input_vals_c, 
                         double *centers_c,
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int threads,
                         int shared_memory_needed);
                         
__global__ void new_centers_shful(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c);
