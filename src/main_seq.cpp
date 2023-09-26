#include <iostream>
#include <chrono>
#include "argparse.h"
#include "io.h"
#include "helper.h"
#include "kmeans_seq.h"

int main(int argc, char **argv){

    // Parse args
    struct options_t opts;
    
    get_opts(argc, argv, &opts);
    
    double *input_vals, *centers;
    int *labels;
    int n_vals;
    read_file(&opts, &n_vals, &input_vals, &labels, &centers);
    struct kmeans_args args;
    fill_kmeans_args(&args, opts.n_cluster, n_vals, opts.dims, opts.max_iter, opts.threshold, input_vals,
                     centers, labels);

    
    random_centers(opts.seed, &args);
    int iter=0;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    //start iteration for computing centroids
    iter = kmeans_cpu(&args);

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("%d,%lf\n", iter, ((double)diff.count()/(iter)));

    output(&args, opts.c_flag);

    free(input_vals);
    free(centers);
    free(labels);
    
    
    return 0;

}