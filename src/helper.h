#ifndef _HELPER_H
#define _HELPER_H
#include "io.h"
#include "argparse.h"

struct kmeans_args{
    int n_cluster;
    int n_vals;
    int dims;
    int max_iter;
    double threshold;
    double *input_vals;
    double *centers;
    int *labels;
};
int kmeans_rand();
void kmeans_srand(unsigned int seed);

void random_centers(int        seed, 
                    int        n_cluster,
                    int        n_vals,
                    int        dim,
                    double*    centers,
                    double*    input_vals);


void fill_kmeans_args(kmeans_args *args,
                      int         n_cluster,
                      int         n_vals,
                      int         dims,
                      int         max_iter,
                      double      threshold,
                      double      *input_vals,
                      double      *centers,
                      int         *labels);
#endif                    