#include "io.h"
#include "argparse.h"

#include "cuda_runtime.h"
#include <math.h>
#include "kmeans_kernel.cuh"

#define THREAD_PER_BLOCK 512

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

bool test_converge(double *centers, double *old_centers, double threshold){
    bool converge = true;
    for (int i = 0; i < sizeof(old_centers); i++){
        //if (old_centers[i] - centers[i] > threshold || old_centers[i] - centers[i] < threshold *(-1)){
        if ( fabs(old_centers[i] - centers[i]) > threshold) {   
            converge = false;
            break;
        }
    }

    return converge;
}

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
    //printf("%d args\n",argc);
    get_opts(argc, argv, &opts);
    //printf("d=%d, k=%d, m=%d, t=%f, s=%d, c=%d\n",opts.dims, opts.n_cluster, opts.max_iter, opts.threshold, opts.seed, opts.c_flag);
    //printf("Last arg is %s", argv[argc-1]);
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

    //struct kmeans_args args;
    //fill_kmeans_args(&args, opts.n_cluster, n_vals, opts.dims, opts.max_iter, opts.threshold, input_vals,
    //                 centers, labels);

    random_centers(opts.seed, opts.n_cluster, n_vals, opts.dims, input_vals, centers);
    //printf("input size:%d\n", input_size);

    double *input_vals_c, *centers_c;
    int *labels_c;
    int *n_points_c;//points count for each centroids
    double *old_centers_c, *temp_centers_c;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t mem_start, mem_stop;
    cudaEventCreate(&mem_start);
    cudaEventCreate(&mem_stop);

    cudaMalloc((double**)&input_vals_c, input_size);
    cudaMalloc((double**)&centers_c, centers_size);
    cudaMalloc((int**)&labels_c, n_vals * sizeof(int));
    cudaMalloc((int**)&n_points_c, opts.n_cluster * sizeof(int));
    cudaMalloc((double**)&old_centers_c, centers_size);
    cudaMalloc((double**)&temp_centers_c, centers_size);
   // printf("input:%lf\n", input_vals[0]);
    cudaMemcpy(centers_c, centers, centers_size, cudaMemcpyHostToDevice);
   // printf("cp centers:%lf\n", centers[0]);
    cudaMemcpy(input_vals_c, input_vals, input_size, cudaMemcpyHostToDevice);
   
    

    //printf("start iter\n");
    int iter = 0;
    float mem_time = 0;
    float temp_time = 0;

    cudaEventRecord(start);
    for (iter = 0; iter < opts.max_iter; iter++){
        cudaEventRecord(mem_start);
        
        cudaMemcpy(old_centers, centers, centers_size, cudaMemcpyHostToHost);
        cudaMemcpy(centers_c, centers, centers_size, cudaMemcpyHostToDevice);
        cudaMemset(temp_centers_c, 0, centers_size);
        cudaMemset(n_points_c, 0, opts.n_cluster*sizeof(int));
        
        cudaEventRecord(mem_stop);
        cudaEventSynchronize(mem_stop);
        cudaEventElapsedTime(&temp_time, mem_start, mem_stop);
        mem_time+= temp_time;
        //printf("mem_time:%lf\n",mem_time);

        /*get_label<<< (n_vals+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>(input_vals_c,
                                                                       centers_c,
                                                                       labels_c,
                                                                       opts.dims,
                                                                       n_vals,
                                                                       opts.n_cluster,
                                                                       temp_centers_c,
                                                                       n_points_c);*/
        wrapper_get_label(input_vals_c, 
                          centers_c,
                          labels_c,
                          opts.dims,
                          n_vals,
                          opts.n_cluster,
                          temp_centers_c,
                          n_points_c,
                          (n_vals+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK,
                          THREAD_PER_BLOCK);
                        
        cudaDeviceSynchronize();
        //printf("after sync\n");

        cudaEventRecord(mem_start);

        cudaMemcpy(temp_centers, temp_centers_c, centers_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, labels_c, n_vals * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(n_points, n_points_c, opts.n_cluster*sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventRecord(mem_stop);
        cudaEventSynchronize(mem_stop);
        cudaEventElapsedTime(&temp_time, mem_start, mem_stop);
        mem_time+= temp_time;
        //printf("n_point: %d\n",n_points[0]);
        

        for(int j = 0; j < opts.n_cluster; j++){
            if(n_points[j]==0)
                continue;
            //printf("temp_centers%d: %lf\n",j,temp_centers[j*opts.dims]);
            for(int d = 0; d< opts.dims; d++){
                
                centers[j*opts.dims+d]=temp_centers[j*opts.dims+d]/n_points[j];
                
            }
        }
        //printf("new centers\n");

        if(test_converge(centers, old_centers, opts.threshold)){
            break;
        }
            
        //printf("centers: %lf\n",centers[0]);
        //printf("centers: %d\n",labels[100]);


    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);

    printf("%d,%lf\n", iter+1, (double)(total_time/(iter+1)));
    printf("data time: %lf\n", mem_time);
    printf("data transfer time fraction: %.2lf%%\n", mem_time/total_time*100);

    output(opts.n_cluster, n_vals, opts.dims, centers, labels, opts.c_flag);

}