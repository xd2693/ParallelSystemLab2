#include "kmeans_kernel.cuh"

__global__ void get_label(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int array_index = index * dims;
    if (index < n_vals){
        
        double distance = DBL_MAX;
        double temp = DBL_MAX;
        for (int i = 0; i < n_cluster; i++){
            double sum=0.0;
            for (int j = 0; j < dims; j++){
                sum+=pow((input_vals_c[array_index+j] - centers_c[i*dims+j]), 2);
            }
            temp = sqrt(sum);
            if (temp < distance){
                distance = temp;
                labels_c[index] = i;
            }
                
                
        }
        

        int center_index = labels_c[index];
        //printf("label:%d\n",center_index);
        
        atomicAdd(&n_points_c[center_index], 1);
        for (int i = 0; i < dims; i++){
            //temp_centers_c[center_index+j]+= input_vals_c[array_index+i];
            atomicAdd(&temp_centers_c[center_index*dims+i], input_vals_c[array_index+i]);  
                  
        }

    } 
}

void wrapper_get_label(double *input_vals_c, 
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
    get_label<<<blocks, threads>>>(input_vals_c,
                                   centers_c,
                                   labels_c,
                                   dims,
                                   n_vals,
                                   n_cluster,
                                   temp_centers_c,
                                   n_points_c);

}

__global__ void get_label_shared(double *input_vals_c, 
                         double *centers_c, 
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c
                         int work_per_thread)
{
    //Calculate shared memory regions, each thread's work starting and end point
    extern __shared__ double s[];
    int center_array_size = n_cluster * dims;
    double * centers_local = s;
    double * new_centers = &s[center_array_size];
    int * n_points_local = (int*)&s[center_array_size * 2];
    int my_local_tid = threadIdx.x;
    int my_global_tid = my_local_tid + blockIdx.x * blockDim.x;
    int my_start_point = my_global_tid * work_per_thread;
    int my_end_point = std::min(n_vals, my_start_point + work_per_thread);
    
    //Prepare shared memory for work to start 
    for (int i = my_local_tid; i < n_cluster; i += blockDim.x){
        n_points_local[i] = 0;
    }
    for (int i = my_local_tid; i < center_array_size; i += blockDim.x) {
        centers_local[i] = centers_c[i];
        new_centers[i] = 0.0;
    }
    __syncthreads();

    //Start the work of adding local copy of center average
    for (int i = my_start_point; i < my_end_point; i ++) {
        //Find owner of each point
        int point_array_start = i * dims;
        double distance_min = DBL_MAX;
        int owner;
        for (int j = 0; j < n_cluster; j++) {
            double sum = 0.0;
            for (int k = 0; k < dims; k++) {
                sum += pow((input_vals_c[point_array_start+k] - centers_local[j*dims+k]), 2);
            }
            if (sum < distance_min) {
                distance_min = sum;
                owner = j;
            }
        }

        //Add the point to owner's local copy
        atomicAdd(&n_points_local[owner], 1);
        for (int i = 0; i < dims; i++) {
            atomicAdd(&new_centers[owner*dims+i], input_vals_c[point_array_start+i]);
        }
    }
    __syncthreads();

    //Add each block's local center into global
    for (int i = my_local_tid; i < n_clusters; i += blockDim.x) {
        atomicAdd(&n_points_c[i], n_points_local[i]);
    }
    for (int i = my_local_tid; i < center_array_size; i += blockDim.x) {
        atomicAdd(&centers_c[i], new_centers[i]);
    }

}

__global__ void wrapper_get_label_shared(double *input_vals_c, 
                         double *centers_c, 
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int blocks,
                         int threads)
{
    int shared_size_needed = 2 * sizeof(double) * dims * n_cluster + sizeof(int) * dims;
    int total_threads = blocks * threads;
    int addition_work = (n_vals % total_threads == 0) ? 0 : 1;
    int work_per_thread = n_vals / total_threads + addition_work;
    get_label_shared<<<blocks, threads, shared_size,needed>>>
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

