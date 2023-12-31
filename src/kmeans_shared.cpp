#include "io.h"
#include "argparse.h"

#include "cuda_runtime.h"
#include <math.h>
#include "kmeans_kernel.cuh"

#define THREAD_PER_BLOCK 64

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

void random_centers(int seed, int n_cluster, int n_vals, int dims, double *input_vals, double *centers) {
    kmeans_srand(seed); 
    int in=0;
    for (int i=0; i<n_cluster; i++){
        int index = (kmeans_rand() % n_vals);
        int my_index= index * dims;
        for (int j=0; j< dims; j++){
            centers[in] = input_vals[my_index+j];
            in++;
        }
        
    }
}

void output(int n_cluster, int n_vals, int dims, double *centers, int *labels, bool c_flag){
    
    if(c_flag){                
        for (int clusterId = 0; clusterId < n_cluster; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < dims; d++)
                printf("%lf ", centers[clusterId * dims + d ]);
            printf("\n");
        }
    }
    else{
        printf("cluster:");
        for (int p=0; p < n_vals; p++)
            printf(" %d", labels[p]);
    }
}

bool test_converge(double *centers, double *old_centers, double threshold, int n_cluster, int dims){

    for (int i = 0; i < n_cluster * dims; i++){
        if ( fabs(old_centers[i] - centers[i]) > threshold) {   
            return false;
        }
    }

    return true;
}

struct time_device{
    cudaEvent_t start, stop;

    float time = 0;
    float temp = 0;
    void start_timing(){
        cudaEventRecord(start);
    }
    void stop_timing(){
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&temp, start, stop);
        time+= temp;
    }
};

int main(int argc, char **argv){

    /*int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);
    printf("Device %d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n", iProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers available per block: %d\n", iProp.regsPerBlock);
    printf("Warp size%d\n", iProp.warpSize);
    printf("Maximum number of threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor/32);
    printf("Device capability %d.%d\n", iProp.major, iProp.minor);*/
    

    // Parse args
    struct options_t opts;
    
    get_opts(argc, argv, &opts);
    
    double *input_vals, *centers;
    double *old_centers, *temp_centers;
    int *labels, *n_points;
    int n_vals;
    
    read_file(&opts, &n_vals, &input_vals, &labels, &centers);

    int input_size = n_vals * opts.dims * sizeof(double);
    int centers_size = opts.n_cluster * opts.dims * sizeof(double);
    old_centers = (double*) malloc(centers_size);
    temp_centers = (double*) malloc(centers_size);
    n_points = (int*) malloc(opts.n_cluster * sizeof(int));
    //set centroids using seed
    random_centers(opts.seed, opts.n_cluster, n_vals, opts.dims, input_vals, centers);

    double *input_vals_c, *centers_c;
    int *labels_c;
    int *n_points_c;//points count for each centroids
    double *old_centers_c, *temp_centers_c;
    
    struct time_device total_time;//total time for data transfer and iteration
    struct time_device mem_time;//data transfer time
    cudaEventCreate(&total_time.start);
    cudaEventCreate(&total_time.stop);
    cudaEventCreate(&mem_time.start);
    cudaEventCreate(&mem_time.stop);

    //Each thread takes 1 datapoint space for input cache
    //Each block holds entire old centers, and local aggregated centers
    //Each block holds entire local aggregated center owned points
    int shared_mem_needed = (THREAD_PER_BLOCK * opts.dims + opts.n_cluster * opts.dims) * sizeof(double);
    bool can_run_shared_mem = (shared_mem_needed <= SHARED_MEMORY_BYTES);
    
    


    cudaMalloc((double**)&input_vals_c, input_size);
    cudaMalloc((double**)&centers_c, centers_size);
    cudaMalloc((int**)&labels_c, n_vals * sizeof(int));
    cudaMalloc((int**)&n_points_c, opts.n_cluster * sizeof(int));
    cudaMalloc((double**)&old_centers_c, centers_size);
    cudaMalloc((double**)&temp_centers_c, centers_size);
    //copy host memory to device memory
    cudaMemcpy(centers_c, centers, centers_size, cudaMemcpyHostToDevice);
   
    cudaMemcpy(input_vals_c, input_vals, input_size, cudaMemcpyHostToDevice);
   
    int iter = 0;
    //set banksize to 8 bytes
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    total_time.start_timing();
    for (iter = 0; iter < opts.max_iter; iter++){
        mem_time.start_timing();

        cudaMemcpy(old_centers, centers, centers_size, cudaMemcpyHostToHost);
        cudaMemcpy(centers_c, centers, centers_size, cudaMemcpyHostToDevice);
        cudaMemset(temp_centers_c, 0, centers_size);
        cudaMemset(n_points_c, 0, opts.n_cluster*sizeof(int));

        mem_time.stop_timing();

        if (can_run_shared_mem) {
            wrapper_new_centers_shful(input_vals_c, 
                              centers_c,
                              labels_c,
                              opts.dims,
                              n_vals,
                              opts.n_cluster,
                              temp_centers_c,
                              n_points_c,
                              THREAD_PER_BLOCK,
                              shared_mem_needed);
        }
        else {
            wrapper_new_centers(input_vals_c, 
                              centers_c,
                              labels_c,
                              opts.dims,
                              n_vals,
                              opts.n_cluster,
                              temp_centers_c,
                              n_points_c,
                              (n_vals+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK,
                              THREAD_PER_BLOCK);
        }

        cudaDeviceSynchronize();

        mem_time.start_timing();

        cudaMemcpy(temp_centers, temp_centers_c, centers_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, labels_c, n_vals * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(n_points, n_points_c, opts.n_cluster*sizeof(int), cudaMemcpyDeviceToHost);
        
        mem_time.stop_timing();

        //devide the sum from the previous on every dimension of the new centroid by the number
        //of points in the centroid group
        for(int j = 0; j < opts.n_cluster; j++){
            if(n_points[j]==0)
                continue;
            
            for(int d = 0; d< opts.dims; d++){
                
                centers[j*opts.dims+d]=temp_centers[j*opts.dims+d]/n_points[j];
                
            }
        }
        

        if(test_converge(centers, old_centers, opts.threshold, opts.n_cluster, opts.dims)){
            iter++;
            break;
        }


    }

    total_time.stop_timing();

    printf("%d,%lf\n", iter, (double)(total_time.time/(iter)));
    //printf("data transter time: %lf\n", mem_time.time);
    //printf("data transfer time fraction: %.2lf%%\n", mem_time.time/total_time.time*100);

    output(opts.n_cluster, n_vals, opts.dims, centers, labels, opts.c_flag);

    free(input_vals);
    free(centers);
    free(labels);
    free(old_centers);
    free(temp_centers);
    free(n_points);
    cudaFree(input_vals_c);
    cudaFree(centers_c);
    cudaFree(labels_c);
    cudaFree(n_points_c);
    cudaFree(old_centers_c);
    cudaFree(temp_centers_c);

}