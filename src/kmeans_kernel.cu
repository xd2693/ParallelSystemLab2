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



__global__ void new_centers_shmem(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c){
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;    
    extern __shared__ double c[];
    int centers_size = n_cluster * dims;    
    double *centers_s = c;    
    int label = 0;
    int dim_interleave = threadIdx.x % dims;

    for (int i = threadIdx.x; i < centers_size; i += blockDim.x) {
        centers_s[i] = centers_c[i];       
    }
    __syncthreads();

    int array_index = index * dims;
    if (index < n_vals){
        
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

__global__ void new_centers_shful(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c){
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ double c[];
    int centers_size = n_cluster * dims;
    double *centers_s = c;
    double *my_input_cache = &c[centers_size + threadIdx.x * dims];
    int label = 0;
    int dim_interleave = threadIdx.x % dims;

    for (int i = threadIdx.x; i < centers_size; i += blockDim.x) {
        centers_s[i] = centers_c[i];       
    }
    __syncthreads();

    int array_index = index * dims;
    if (index < n_vals){
        for (int i = 0; i < dims; i++){
            int interleaved_index = (dim_interleave + i) % dims;
            my_input_cache[interleaved_index ] = input_vals_c[array_index + interleaved_index];
        }
        double distance = DBL_MAX;
        double sum;
        for (int i = 0; i < n_cluster; i++){
            sum = 0;
            for (int j = 0; j < dims; j++){
                int interleaved_index = (dim_interleave + j) % dims;
                double d1 = my_input_cache[interleaved_index] - centers_s[i*dims+interleaved_index];
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
            int interleaved_index = (dim_interleave + i) % dims;
            atomicAdd(&temp_centers_c[label*dims+interleaved_index], my_input_cache[interleaved_index]);  

        }

    }
     
}

void wrapper_new_centers_shful(double *input_vals_c, 
                         double *centers_c,
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int threads,
                         int shared_memory_needed)
{
    int blocks = (n_vals + threads -1) / threads;
    new_centers_shful<<<blocks, threads, shared_memory_needed>>>
                    (input_vals_c,
                     centers_c,
                     labels_c,
                     dims,
                     n_vals,
                     n_cluster,
                     temp_centers_c,
                     n_points_c);
}
