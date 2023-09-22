#include "io.h"
#include "argparse.h"
#include "helper.h"
#include <cuda_runtime.h>

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

    random_centers(opts.seed, &args);

    double *input_vals_c, *centers_c;
    int *labels_c;
    int *n_points;//points count for each centroids
    double *old_centers_c, temp_centers_c;
    cudaMalloc((double**)&input_vals_c, n_vals * opts.dims * sizeof(double));
    cudaMalloc((double**)&centers_c, opts.n_cluster * opts.dims * sizeof(double));
    cudaMalloc((int**)&labels_c, opts.n_cluster * sizeof(int));
    cudaMalloc((int**)&n_points, opts.n_cluster * sizeof(int));

    cudaMemcpy(input_vals_c, input_vals, cudaMemcpyHostToDevice);
    cudaMemcpy(centers_c, centers, cudaMemcpyHostToDevice);

    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);
    printf("Device %d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n",
    iProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block: %4.2f KB\n",
    iProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers available per block: %d\n",
    iProp.regsPerBlock);
    printf("Warp size%d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n",
    iProp.maxThreadsPerMultiProcessor/32);


}