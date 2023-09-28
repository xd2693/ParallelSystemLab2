#include "kmeans_kernel.cuh"

__global__ void new_centers(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int array_index = index * dims;
    int label = 0;
    if (index < n_vals){
        
        double distance = DBL_MAX;
        double sum;
        for (int i = 0; i < n_cluster; i++){
            sum = 0;
            for (int j = 0; j < dims; j++){
                double d1 = input_vals_c[array_index+j] - centers_c[i*dims+j];
                sum += (d1 * d1);
            }
            //temp = sqrt(sum);
            if (sum < distance){
                distance = sum;
                label = i;
            }
                
                
        }
        labels_c[index] = label;
        //add 1 to number of points in the centroid group 
        atomicAdd(&n_points_c[label], 1);
        for (int i = 0; i < dims; i++){
            //add every dimension to the new centroid for average later
            atomicAdd(&temp_centers_c[label*dims+i], input_vals_c[array_index+i]);  
                  
        }

    } 
}

void wrapper_new_centers(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int blocks,
                         int threads)
{
    new_centers<<<blocks, threads>>>(input_vals_c,
                                   centers_c,
                                   labels_c,
                                   dims,
                                   n_vals,
                                   n_cluster,
                                   temp_centers_c,
                                   n_points_c);

}

__global__ void new_centers_shared(double *input_vals_c, 
                         double *centers_c,
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int work_per_thread)
{
    //Calculate shared memory regions, each thread's work starting and end point
    extern __shared__ double s[];
    int center_array_size = n_cluster * dims;
    double * centers_local = s;
    //double * new_centers = &s[center_array_size];
    //double * my_input_cache = &s[center_array_size * 2 + threadIdx.x * dims];
    double * my_input_cache = &s[center_array_size + threadIdx.x * dims];
    //int * n_points_local = (int*)&s[center_array_size * 2 + blockDim.x * dims];
    int my_global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int my_start_point = my_global_tid * work_per_thread;
    int my_end_point = min(n_vals, my_start_point + work_per_thread);
    int dim_interleave = threadIdx.x % dims;
    
    //Prepare shared memory for work to start 
    /*
    for (int i = threadIdx.x; i < n_cluster; i += blockDim.x){
        n_points_local[i] = 0;
    }*/
    for (int i = threadIdx.x; i < center_array_size; i += blockDim.x) {
        centers_local[i] = centers_c[i];
        //new_centers[i] = 0.0;
    }
    __syncthreads();

    //Start the work of adding local copy of center average
    for (int i = my_start_point; i < my_end_point; i ++) {
        //Find owner of each point
        int point_array_start = i * dims;
        double distance_min = DBL_MAX;
        int owner;
        //Copy input into local cache
        for (int j = 0; j < dims; j++) {
            int interleaved_index = (dim_interleave + j) % dims;
            my_input_cache[interleaved_index] = input_vals_c[point_array_start + interleaved_index];
        }
        for (int j = 0; j < n_cluster; j++) {
            double sum = 0.0;
            for (int k = 0; k < dims; k++) {
                int interleaved_index = (dim_interleave + k) % dims;
                double d1 = my_input_cache[interleaved_index] - centers_local[j*dims+interleaved_index];
                sum += (d1 * d1);
            }
            if (sum < distance_min) {
                distance_min = sum;
                owner = j;
            }
        }
        labels_c[i] = owner;
        //Add the point to owner's local copy
        atomicAdd(&n_points_c[owner], 1);
        for (int j = 0; j < dims; j++) {
            int interleaved_index = (dim_interleave + j) % dims;
            atomicAdd(&temp_centers_c[owner*dims+interleaved_index], my_input_cache[interleaved_index]);
        }
    }
    /*
    __syncthreads();

    //Add each block's local center into global
    for (int i = threadIdx.x; i < n_cluster; i += blockDim.x) {
        atomicAdd(&n_points_c[i], n_points_local[i]);
    }
    for (int i = threadIdx.x; i < center_array_size; i += blockDim.x) {
        atomicAdd(&temp_centers_c[i], new_centers[i]);
    }*/

}

void wrapper_new_centers_shared(double *input_vals_c, 
                         double *centers_c,
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int blocks,
                         int threads,
                         int shared_mem)
{
    int total_threads = blocks * threads;
    int addition_work = (n_vals % total_threads == 0) ? 0 : 1;
    int work_per_thread = n_vals / total_threads + addition_work;
    new_centers_shared<<<blocks, threads, shared_mem>>>
                    (input_vals_c,
                     centers_c,
                     labels_c,
                     dims,
                     n_vals,
                     n_cluster,
                     temp_centers_c,
                     n_points_c,
                     work_per_thread);
}

__global__ void new_centers_shmem(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c){
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //int input_index = index * dims;
    extern __shared__ double c[];
    int centers_size = n_cluster * dims;
    //int input_size = n_vals * dims;
    double *centers_s = c;
    //don't put input into shared memory
    //double *input_s = &c[centers_size];
    int label = 0;
    int dim_interleave = threadIdx.x % dims;

    for (int i = threadIdx.x; i < centers_size; i += blockDim.x) {
        centers_s[i] = centers_c[i];       
    }
    __syncthreads();

    int array_index = index * dims;
    if (index < n_vals){
        /*
        for (int i = 0; i < dims; i++){
            int interleaved_index = (dim_interleave + i) % dims;
            input_s[array_index + interleaved_index ] = input_vals_c[index * dims + interleaved_index];
            //input_vals_c[index * dims + i] = threadIdx.x * dims + i;
        }
        */
        double distance = DBL_MAX;
        double sum;
        for (int i = 0; i < n_cluster; i++){
            sum = 0;
            for (int j = 0; j < dims; j++){
                int interleaved_index = (dim_interleave + j) % dims;
                double d1 = input_vals_c[array_index+interleaved_index] - centers_s[i*dims+interleaved_index];
                sum += (d1 * d1);
            }
            
            if (sum < distance){
                distance = sum;
                label = i;
            }
                
                
        }
        labels_c[index] = label;

        
        
        atomicAdd(&n_points_c[label], 1);
        for (int i = 0; i < dims; i++){
            //temp_centers_c[center_index+j]+= input_vals_c[array_index+i];
            int interleaved_index = (dim_interleave + i) % dims;
            atomicAdd(&temp_centers_c[label*dims+interleaved_index], input_vals_c[array_index+interleaved_index]);  
              
        }

    }
     
}

void wrapper_new_centers_shmem(double *input_vals_c, 
                         double *centers_c,
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int threads)
{
    int shared_size_needed = sizeof(double) * dims * n_cluster;
    int blocks = (n_vals + threads -1) / threads;
    new_centers_shmem<<<blocks, threads, shared_size_needed>>>
                    (input_vals_c,
                     centers_c,
                     labels_c,
                     dims,
                     n_vals,
                     n_cluster,
                     temp_centers_c,
                     n_points_c);
}

