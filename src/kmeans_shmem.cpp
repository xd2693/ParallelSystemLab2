#include "io.h"
#include "argparse.h"

#include "cuda_runtime.h"
#include <math.h>
#include "kmeans_kernel.cuh"

#define THREAD_PER_BLOCK 128

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
        //printf("\n index= %d\n",index);
        for (int j=0; j< dims; j++){
            centers[in] = input_vals[my_index+j];
            //printf("\t centers= %.12f",args->centers[in]);
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
        //if (old_centers[i] - centers[i] > threshold || old_centers[i] - centers[i] < threshold *(-1)){
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
    /*
    int iDev = 0;
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
    int threads = 128, blocks = 40;
    
    
    struct time_device total_time;//total time for data transfer and iteration
    struct time_device mem_time;//data transfer time
    struct time_device process_time;
    cudaEventCreate(&total_time.start);
    cudaEventCreate(&total_time.stop);
    cudaEventCreate(&mem_time.start);
    cudaEventCreate(&mem_time.stop);
    cudaEventCreate(&process_time.start);
    cudaEventCreate(&process_time.stop);
    
    //printf("Setting mem bank unit to 8B\n");
    //set banksize to 8 bytes
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    total_time.start_timing();
    mem_time.start_timing();

    cudaMalloc((double**)&input_vals_c, input_size);
    cudaMalloc((double**)&centers_c, centers_size);
    cudaMalloc((int**)&labels_c, n_vals * sizeof(int));
    cudaMalloc((int**)&n_points_c, opts.n_cluster * sizeof(int));
    cudaMalloc((double**)&old_centers_c, centers_size);
    cudaMalloc((double**)&temp_centers_c, centers_size);
    //copy host memory to device memory
    cudaMemcpy(centers_c, centers, centers_size, cudaMemcpyHostToDevice);
   
    cudaMemcpy(input_vals_c, input_vals, input_size, cudaMemcpyHostToDevice);
   
    mem_time.stop_timing();
    
    
    int iter = 0;
           
    for (iter = 0; iter < opts.max_iter; iter++){
        mem_time.start_timing();

        cudaMemcpy(old_centers, centers, centers_size, cudaMemcpyHostToHost);
        cudaMemcpy(centers_c, centers, centers_size, cudaMemcpyHostToDevice);
        cudaMemset(temp_centers_c, 0, centers_size);
        cudaMemset(n_points_c, 0, opts.n_cluster*sizeof(int));

        mem_time.stop_timing();
          
        process_time.start_timing();
        
        wrapper_get_label_shmem(input_vals_c, 
                          centers_c,
                          labels_c,
                          opts.dims,
                          n_vals,
                          opts.n_cluster,
                          temp_centers_c,
                          n_points_c,
                          THREAD_PER_BLOCK);

        cudaDeviceSynchronize();
        process_time.stop_timing();
        
        
        mem_time.start_timing();

        cudaMemcpy(temp_centers, temp_centers_c, centers_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, labels_c, n_vals * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(n_points, n_points_c, opts.n_cluster*sizeof(int), cudaMemcpyDeviceToHost);
        
        mem_time.stop_timing();

        for(int j = 0; j < opts.n_cluster; j++){
            if(n_points[j]==0)
                continue;
            //printf("temp_centers%d: %lf\n",j,temp_centers[j*opts.dims]);
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

    printf("Running with %d blocks %d threads\n", blocks, threads);
    printf("%d,%lf\n", iter, (double)(total_time.time/(iter)));
    printf("data transter time: %lf\n", mem_time.time);
    printf("data transfer time fraction: %.2lf%%\n", mem_time.time/total_time.time*100);
    printf("Pure process time per step %lf\n", (double)(process_time.time/iter));

    output(opts.n_cluster, n_vals, opts.dims, centers, labels, opts.c_flag);

}