#include <iostream>
#include <chrono>
#include "argparse.h"
#include "io.h"
#include "helper.h"
#include "kmeans.h"

int main(int argc, char **argv){

    // Parse args
    struct options_t opts;
    //printf("%d args\n",argc);
    get_opts(argc, argv, &opts);
    //printf("d=%d, k=%d, m=%d, t=%f, s=%d, c=%d\n",opts.dims, opts.n_cluster, opts.max_iter, opts.threshold, opts.seed, opts.c_flag);
    //printf("Last arg is %s", argv[argc-1]);
    double *input_vals, *centers;
    int *labels;
    int n_vals;
    read_file(&opts, &n_vals, &input_vals, &labels, &centers);
    struct kmeans_args args;
    fill_kmeans_args(&args, opts.n_cluster, n_vals, opts.dims, opts.max_iter, opts.threshold, input_vals,
                     centers, labels);

    //printf("%12f ",input_vals[100]);
    random_centers(opts.seed, &args);
    int iter=0;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    /*timing_device total_time;
    total_time.start_timing();*/

    iter = kmeans_cpu(&args);

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //total_time.end_timing();

    printf("%d,%lf\n", iter+1, (double)(diff.count()/(iter+1)));

    output(&args, opts.c_flag);

    
    return 0;

}