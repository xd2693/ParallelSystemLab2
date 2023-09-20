#ifndef _HELPER_H
#define _HELPER_H
#include "io.h"
#include "argparse.h"

int kmeans_rand();
void kmeans_srand(unsigned int seed);

void random_centers(int        seed, 
                    int        n_cluster,
                    int        n_vals,
                    int        dim,
                    double*    centers,
                    double*    input_vals);
#endif                    